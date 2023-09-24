_M='hijack'
_L='move model to device'
_K='find config'
_J='loaded_vae_file'
_I='base_vae'
_H='sd_checkpoint_hash'
_G='conditioner'
_F='calculate hash'
_E='sd_model_checkpoint'
_D='.safetensors'
_C=True
_B=False
_A=None
import collections,os.path,sys,gc,threading,torch,re,safetensors.torch
from omegaconf import OmegaConf
from os import mkdir
from urllib import request
import ldm.modules.midas as midas
from ldm.util import instantiate_from_config
from modules import paths,shared,modelloader,devices,script_callbacks,sd_vae,sd_disable_initialization,errors,hashes,sd_models_config,sd_unet,sd_models_xl,cache,extra_networks,processing,lowvram,sd_hijack
from modules.timer import Timer
import tomesd
model_dir='Standard-demo'
model_path=os.path.abspath(os.path.join(paths.models_path,model_dir))
checkpoints_list={}
checkpoint_aliases={}
checkpoint_alisases=checkpoint_aliases
checkpoints_loaded=collections.OrderedDict()
def replace_key(d,key,new_key,value):
	B=new_key;A=list(d.keys());d[B]=value
	if key not in A:return d
	C=A.index(key);A[C]=B;D={A:d[A]for A in A};d.clear();d.update(D);return d
class CheckpointInfo:
	def __init__(A,filename):
		C=filename;A.filename=C;D=os.path.abspath(C);A.is_safetensors=os.path.splitext(C)[1].lower()==_D
		if shared.cmd_opts.ckpt_dir is not _A and D.startswith(shared.cmd_opts.ckpt_dir):B=D.replace(shared.cmd_opts.ckpt_dir,'')
		elif D.startswith(model_path):B=D.replace(model_path,'')
		else:B=os.path.basename(C)
		if B.startswith('\\')or B.startswith('/'):B=B[1:]
		def E():B=read_metadata_from_safetensors(C);A.modelspec_thumbnail=B.pop('modelspec.thumbnail',_A);return B
		A.metadata={}
		if A.is_safetensors:
			try:A.metadata=cache.cached_data_for_file('safetensors-metadata','checkpoint/'+B,C,E)
			except Exception as F:errors.display(F,f"reading metadata for {C}")
		A.name=B;A.name_for_extra=os.path.splitext(os.path.basename(C))[0];A.model_name=os.path.splitext(B.replace('/','_').replace('\\','_'))[0];A.hash=model_hash(C);A.sha256=hashes.sha256_from_cache(A.filename,f"checkpoint/{B}");A.shorthash=A.sha256[0:10]if A.sha256 else _A;A.title=B if A.shorthash is _A else f"{B} [{A.shorthash}]";A.short_title=A.name_for_extra if A.shorthash is _A else f"{A.name_for_extra} [{A.shorthash}]";A.ids=[A.hash,A.model_name,A.title,B,A.name_for_extra,f"{B} [{A.hash}]"]
		if A.shorthash:A.ids+=[A.shorthash,A.sha256,f"{A.name} [{A.shorthash}]",f"{A.name_for_extra} [{A.shorthash}]"]
	def register(A):
		checkpoints_list[A.title]=A
		for id in A.ids:checkpoint_aliases[id]=A
	def calculate_shorthash(A):
		A.sha256=hashes.sha256(A.filename,f"checkpoint/{A.name}")
		if A.sha256 is _A:return
		B=A.sha256[0:10]
		if A.shorthash==A.sha256[0:10]:return A.shorthash
		A.shorthash=B
		if A.shorthash not in A.ids:A.ids+=[A.shorthash,A.sha256,f"{A.name} [{A.shorthash}]",f"{A.name_for_extra} [{A.shorthash}]"]
		C=A.title;A.title=f"{A.name} [{A.shorthash}]";A.short_title=f"{A.name_for_extra} [{A.shorthash}]";replace_key(checkpoints_list,C,A.title,A);A.register();return A.shorthash
try:from transformers import logging,CLIPModel;logging.set_verbosity_error()
except Exception:pass
def setup_model():os.makedirs(model_path,exist_ok=_C);enable_midas_autodownload()
def checkpoint_tiles(use_short=_B):return[A.short_title if use_short else A.title for A in checkpoints_list.values()]
def list_models():
	checkpoints_list.clear();checkpoint_aliases.clear();A=shared.cmd_opts.ckpt
	if shared.cmd_opts.no_download_sd_model or A!=shared.sd_model_file or os.path.exists(A):C=_A
	else:C='https://huggingface.co/runwayml/standard-demo-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors'
	D=modelloader.load_models(model_path=model_path,model_url=C,command_path=shared.cmd_opts.ckpt_dir,ext_filter=['.ckpt',_D],download_name='v1-5-pruned-emaonly.safetensors',ext_blacklist=['.vae.ckpt','.vae.safetensors'])
	if os.path.exists(A):B=CheckpointInfo(A);B.register();shared.opts.data[_E]=B.title
	elif A is not _A and A!=shared.default_sd_model_file:print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {A}",file=sys.stderr)
	for E in D:B=CheckpointInfo(E);B.register()
re_strip_checksum=re.compile('\\s*\\[[^]]+]\\s*$')
def get_closet_checkpoint_match(search_string):
	B=search_string
	if not B:return
	C=checkpoint_aliases.get(B,_A)
	if C is not _A:return C
	A=sorted([A for A in checkpoints_list.values()if B in A.title],key=lambda x:len(x.title))
	if A:return A[0]
	D=re.sub(re_strip_checksum,'',B);A=sorted([A for A in checkpoints_list.values()if D in A.title],key=lambda x:len(x.title))
	if A:return A[0]
def model_hash(filename):
	'old hash that only looks at a small part of the file and is prone to collisions'
	try:
		with open(filename,'rb')as A:import hashlib as C;B=C.sha256();A.seek(1048576);B.update(A.read(65536));return B.hexdigest()[0:8]
	except FileNotFoundError:return'NOFILE'
def select_checkpoint():
	'Raises `FileNotFoundError` if no checkpoints are found.';C=shared.opts.sd_model_checkpoint;A=checkpoint_aliases.get(C,_A)
	if A is not _A:return A
	if len(checkpoints_list)==0:
		B='No checkpoints found. When searching for checkpoints, looked at:'
		if shared.cmd_opts.ckpt is not _A:B+=f"\n - file {os.path.abspath(shared.cmd_opts.ckpt)}"
		B+=f"\n - directory {model_path}"
		if shared.cmd_opts.ckpt_dir is not _A:B+=f"\n - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}"
		B+="Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations.";raise FileNotFoundError(B)
	A=next(iter(checkpoints_list.values()))
	if C is not _A:print(f"Checkpoint {C} not found; loading fallback {A.title}",file=sys.stderr)
	return A
checkpoint_dict_replacements={'cond_stage_model.transformer.embeddings.':'cond_stage_model.transformer.text_model.embeddings.','cond_stage_model.transformer.encoder.':'cond_stage_model.transformer.text_model.encoder.','cond_stage_model.transformer.final_layer_norm.':'cond_stage_model.transformer.text_model.final_layer_norm.'}
def transform_checkpoint_dict_key(k):
	for(A,B)in checkpoint_dict_replacements.items():
		if k.startswith(A):k=B+k[len(A):]
	return k
def get_state_dict_from_checkpoint(pl_sd):
	D='state_dict';A=pl_sd;A=A.pop(D,A);A.pop(D,_A);B={}
	for(E,F)in A.items():
		C=transform_checkpoint_dict_key(E)
		if C is not _A:B[C]=F
	A.clear();A.update(B);return A
def read_metadata_from_safetensors(filename):
	E=filename;import json as F
	with open(E,mode='rb')as C:
		A=C.read(8);A=int.from_bytes(A,'little');G=C.read(2);assert A>2 and G in(b'{"',b"{'"),f"{E} is not a safetensors file";I=G+C.read(A-2);J=F.loads(I);D={}
		for(H,B)in J.get('__metadata__',{}).items():
			D[H]=B
			if isinstance(B,str)and B[0:1]=='{':
				try:D[H]=F.loads(B)
				except Exception:pass
		return D
def read_state_dict(checkpoint_file,print_global_state=_B,map_location=_A):
	E='global_step';C=map_location;B=checkpoint_file;H,F=os.path.splitext(B)
	if F.lower()==_D:
		D=C or shared.weight_load_location or devices.get_optimal_device_name()
		if not shared.opts.disable_mmap_load_safetensors:A=safetensors.torch.load_file(B,device=D)
		else:A=safetensors.torch.load(open(B,'rb').read());A={A:B.to(D)for(A,B)in A.items()}
	else:A=torch.load(B,map_location=C or shared.weight_load_location)
	if print_global_state and E in A:print(f"Global Step: {A[E]}")
	G=get_state_dict_from_checkpoint(A);return G
def get_checkpoint_state_dict(checkpoint_info,timer):
	B=timer;A=checkpoint_info;C=A.calculate_shorthash();B.record(_F)
	if A in checkpoints_loaded:print(f"Loading weights [{C}] from cache");return checkpoints_loaded[A]
	print(f"Loading weights [{C}] from {A.filename}");D=read_state_dict(A.filename);B.record('load weights from disk');return D
class SkipWritingToConfig:
	'This context manager prevents load_model_weights from writing checkpoint name to the config when it loads weight.';skip=_B;previous=_A
	def __enter__(A):A.previous=SkipWritingToConfig.skip;SkipWritingToConfig.skip=_C;return A
	def __exit__(A,exc_type,exc_value,exc_traceback):SkipWritingToConfig.skip=A.previous
def load_model_weights(model,checkpoint_info,state_dict,timer):
	D=state_dict;C=timer;B=checkpoint_info;A=model;F=B.calculate_shorthash();C.record(_F)
	if not SkipWritingToConfig.skip:shared.opts.data[_E]=B.title
	if D is _A:D=get_checkpoint_state_dict(B,C)
	A.is_sdxl=hasattr(A,_G);A.is_sd2=not A.is_sdxl and hasattr(A.cond_stage_model,'model');A.is_sd1=not A.is_sdxl and not A.is_sd2
	if A.is_sdxl:sd_models_xl.extend_sdxl(A)
	A.load_state_dict(D,strict=_B);C.record('apply weights to model')
	if shared.opts.sd_checkpoint_cache>0:checkpoints_loaded[B]=D
	del D
	if shared.cmd_opts.opt_channelslast:A.to(memory_format=torch.channels_last);C.record('apply channels_last')
	if shared.cmd_opts.no_half:A.float();devices.dtype_unet=torch.float32;C.record('apply float()')
	else:
		G=A.first_stage_model;E=getattr(A,'depth_model',_A)
		if shared.cmd_opts.no_half_vae:A.first_stage_model=_A
		if shared.cmd_opts.upcast_sampling and E:A.depth_model=_A
		A.half();A.first_stage_model=G
		if E:A.depth_model=E
		devices.dtype_unet=torch.float16;C.record('apply half()')
	devices.unet_needs_upcast=shared.cmd_opts.upcast_sampling and devices.dtype==torch.float16 and devices.dtype_unet==torch.float16;A.first_stage_model.to(devices.dtype_vae);C.record('apply dtype to VAE')
	while len(checkpoints_loaded)>shared.opts.sd_checkpoint_cache:checkpoints_loaded.popitem(last=_B)
	A.sd_model_hash=F;A.sd_model_checkpoint=B.filename;A.sd_checkpoint_info=B;shared.opts.data[_H]=B.sha256
	if hasattr(A,'logvar'):A.logvar=A.logvar.to(devices.device)
	sd_vae.delete_base_vae();sd_vae.clear_loaded_vae();H,I=sd_vae.resolve_vae(B.filename).tuple();sd_vae.load_vae(A,H,I);C.record('load VAE')
def enable_midas_autodownload():
	'\n    Gives the ldm.modules.midas.api.load_model function automatic downloading.\n\n    When the 512-depth-ema model, and other future models like it, is loaded,\n    it calls midas.api.load_model to load the associated midas depth model.\n    This function applies a wrapper to download the model to the correct\n    location automatically.\n    ';B=os.path.join(paths.models_path,'midas')
	for(A,C)in midas.api.ISL_PATHS.items():D=os.path.basename(C);midas.api.ISL_PATHS[A]=os.path.join(B,D)
	E={'dpt_large':'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt','dpt_hybrid':'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt','midas_v21':'https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt','midas_v21_small':'https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt'};midas.api.load_model_inner=midas.api.load_model
	def F(model_type):
		A=model_type;C=midas.api.ISL_PATHS[A]
		if not os.path.exists(C):
			if not os.path.exists(B):mkdir(B)
			print(f"Downloading midas model weights for {A} to {C}");request.urlretrieve(E[A],C);print(f"{A} downloaded")
		return midas.api.load_model_inner(A)
	midas.api.load_model=F
def repair_config(sd_config):
	A=sd_config
	if not hasattr(A.model.params,'use_ema'):A.model.params.use_ema=_B
	if hasattr(A.model.params,'unet_config'):
		if shared.cmd_opts.no_half:A.model.params.unet_config.params.use_fp16=_B
		elif shared.cmd_opts.upcast_sampling:A.model.params.unet_config.params.use_fp16=_C
	if getattr(A.model.params.first_stage_config.params.ddconfig,'attn_type',_A)=='vanilla-xformers'and not shared.xformers_available:A.model.params.first_stage_config.params.ddconfig.attn_type='vanilla'
	if hasattr(A.model.params,'noise_aug_config')and hasattr(A.model.params.noise_aug_config.params,'clip_stats_path'):B=os.path.join(paths.models_path,'karlo');A.model.params.noise_aug_config.params.clip_stats_path=A.model.params.noise_aug_config.params.clip_stats_path.replace('checkpoints/karlo_models',B)
sd1_clip_weight='cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight='cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'
sdxl_clip_weight='conditioner.embedders.1.model.ln_final.weight'
sdxl_refiner_clip_weight='conditioner.embedders.0.model.ln_final.weight'
class SdModelData:
	def __init__(A):A.sd_model=_A;A.loaded_sd_models=[];A.was_loaded_at_least_once=_B;A.lock=threading.Lock()
	def get_sd_model(A):
		if A.was_loaded_at_least_once:return A.sd_model
		if A.sd_model is _A:
			with A.lock:
				if A.sd_model is not _A or A.was_loaded_at_least_once:return A.sd_model
				try:load_model()
				except Exception as B:errors.display(B,'loading standard demo model',full_traceback=_C);print('',file=sys.stderr);print('Standard demo model failed to load',file=sys.stderr);A.sd_model=_A
		return A.sd_model
	def set_sd_model(A,v,already_loaded=_B):
		A.sd_model=v
		if already_loaded:sd_vae.base_vae=getattr(v,_I,_A);sd_vae.loaded_vae_file=getattr(v,_J,_A);sd_vae.checkpoint_info=v.sd_checkpoint_info
		try:A.loaded_sd_models.remove(v)
		except ValueError:pass
		if v is not _A:A.loaded_sd_models.insert(0,v)
model_data=SdModelData()
def get_empty_cond(sd_model):
	A=sd_model;B=processing.StandardDemoProcessingTxt2Img();extra_networks.activate(B,{})
	if hasattr(A,_G):C=A.get_learned_conditioning(['']);return C['crossattn']
	else:return A.cond_stage_model([''])
def send_model_to_cpu(m):
	if m.lowvram:lowvram.send_everything_to_cpu()
	else:m.to(devices.cpu)
	devices.torch_gc()
def model_target_device(m):
	if lowvram.is_needed(m):return devices.cpu
	else:return devices.device
def send_model_to_device(m):
	lowvram.apply(m)
	if not m.lowvram:m.to(shared.device)
def send_model_to_trash(m):m.to(device='meta');devices.torch_gc()
def load_model(checkpoint_info=_A,already_loaded_state_dict=_A):
	G=already_loaded_state_dict;C=checkpoint_info;from modules import sd_hijack as H;C=C or select_checkpoint();B=Timer()
	if model_data.sd_model:send_model_to_trash(model_data.sd_model);model_data.sd_model=_A;devices.torch_gc()
	B.record('unload existing model')
	if G is not _A:D=G
	else:D=get_checkpoint_state_dict(C,B)
	E=sd_models_config.find_checkpoint_config(D,C);J=any(A for A in[sd1_clip_weight,sd2_clip_weight,sdxl_clip_weight,sdxl_refiner_clip_weight]if A in D);B.record(_K);F=OmegaConf.load(E);repair_config(F);B.record('load config');print(f"Creating model from config: {E}");A=_A
	try:
		with sd_disable_initialization.DisableInitialization(disable_clip=J or shared.cmd_opts.do_not_download_clip):
			with sd_disable_initialization.InitializeOnMeta():A=instantiate_from_config(F.model)
	except Exception as K:errors.display(K,'creating model quickly',full_traceback=_C)
	if A is _A:
		print('Failed to create model quickly; will retry using slow method.',file=sys.stderr)
		with sd_disable_initialization.InitializeOnMeta():A=instantiate_from_config(F.model)
	A.used_config=E;B.record('create model')
	if shared.cmd_opts.no_half:I=_A
	else:I={'first_stage_model':_A,'':torch.float16}
	with sd_disable_initialization.LoadStateDictOnMeta(D,device=model_target_device(A),weight_dtype_conversion=I):load_model_weights(A,C,D,B)
	B.record('load weights from state dict');send_model_to_device(A);B.record(_L);H.model_hijack.hijack(A);B.record(_M);A.eval();model_data.set_sd_model(A);model_data.was_loaded_at_least_once=_C;H.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=_C);B.record('load textual inversion embeddings');script_callbacks.model_loaded_callback(A);B.record('scripts callbacks')
	with devices.autocast(),torch.no_grad():A.cond_stage_model_empty_prompt=get_empty_cond(A)
	B.record('calculate empty prompt');print(f"Model loaded in {B.summary()}.");return A
def reuse_model_from_already_loaded(sd_model,checkpoint_info,timer):
	"\n    Checks if the desired checkpoint from checkpoint_info is not already loaded in model_data.loaded_sd_models.\n    If it is loaded, returns that (moving it to GPU if necessary, and moving the currently loadded model to CPU if necessary).\n    If not, returns the model that can be used to load weights from checkpoint_info's file.\n    If no such model exists, returns None.\n    Additionaly deletes loaded models that are over the limit set in settings (sd_checkpoints_limit).\n    ";D=timer;C=checkpoint_info;A=sd_model;B=_A
	for F in reversed(range(len(model_data.loaded_sd_models))):
		E=model_data.loaded_sd_models[F]
		if E.sd_checkpoint_info.filename==C.filename:B=E;continue
		if len(model_data.loaded_sd_models)>shared.opts.sd_checkpoints_limit>0:print(f"Unloading model {len(model_data.loaded_sd_models)} over the limit of {shared.opts.sd_checkpoints_limit}: {E.sd_checkpoint_info.title}");model_data.loaded_sd_models.pop();send_model_to_trash(E);D.record('send model to trash')
		if shared.opts.sd_checkpoints_keep_in_cpu:send_model_to_cpu(A);D.record('send model to cpu')
	if B is not _A:
		send_model_to_device(B);D.record('send model to device');model_data.set_sd_model(B,already_loaded=_C)
		if not SkipWritingToConfig.skip:shared.opts.data[_E]=B.sd_checkpoint_info.title;shared.opts.data[_H]=B.sd_checkpoint_info.sha256
		print(f"Using already loaded model {B.sd_checkpoint_info.title}: done in {D.summary()}");sd_vae.reload_vae_weights(B);return model_data.sd_model
	elif shared.opts.sd_checkpoints_limit>1 and len(model_data.loaded_sd_models)<shared.opts.sd_checkpoints_limit:print(f"Loading model {C.title} ({len(model_data.loaded_sd_models)+1} out of {shared.opts.sd_checkpoints_limit})");model_data.sd_model=_A;load_model(C);return model_data.sd_model
	elif len(model_data.loaded_sd_models)>0:A=model_data.loaded_sd_models.pop();model_data.sd_model=A;sd_vae.base_vae=getattr(A,_I,_A);sd_vae.loaded_vae_file=getattr(A,_J,_A);sd_vae.checkpoint_info=A.sd_checkpoint_info;print(f"Reusing loaded model {A.sd_checkpoint_info.title} to load {C.title}");return A
	else:return
def reload_model_weights(sd_model=_A,info=_A):
	A=sd_model;C=info or select_checkpoint();B=Timer()
	if not A:A=model_data.sd_model
	if A is _A:E=_A
	else:
		E=A.sd_checkpoint_info
		if A.sd_model_checkpoint==C.filename:return A
	A=reuse_model_from_already_loaded(A,C,B)
	if A is not _A and A.sd_checkpoint_info.filename==C.filename:return A
	if A is not _A:sd_unet.apply_unet('None');send_model_to_cpu(A);sd_hijack.model_hijack.undo_hijack(A)
	D=get_checkpoint_state_dict(C,B);F=sd_models_config.find_checkpoint_config(D,C);B.record(_K)
	if A is _A or F!=A.used_config:
		if A is not _A:send_model_to_trash(A)
		load_model(C,already_loaded_state_dict=D);return model_data.sd_model
	try:load_model_weights(A,C,D,B)
	except Exception:print('Failed to load checkpoint, restoring previous');load_model_weights(A,E,_A,B);raise
	finally:
		sd_hijack.model_hijack.hijack(A);B.record(_M);script_callbacks.model_loaded_callback(A);B.record('script callbacks')
		if not A.lowvram:A.to(devices.device);B.record(_L)
	print(f"Weights loaded in {B.summary()}.");model_data.set_sd_model(A);sd_unet.apply_unet();return A
def unload_model_weights(sd_model=_A,info=_A):
	A=sd_model;B=Timer()
	if model_data.sd_model:model_data.sd_model.to(devices.cpu);sd_hijack.model_hijack.undo_hijack(model_data.sd_model);model_data.sd_model=_A;A=_A;gc.collect();devices.torch_gc()
	print(f"Unloaded weights {B.summary()}.");return A
def apply_token_merging(sd_model,token_merging_ratio):
	'\n    Applies speed and memory optimizations from tomesd.\n    ';B=token_merging_ratio;A=sd_model;C=getattr(A,'applied_token_merged_ratio',0)
	if C==B:return
	if C>0:tomesd.remove_patch(A)
	if B>0:tomesd.apply_patch(A,ratio=B,use_rand=_B,merge_attn=_C,merge_crossattn=_B,merge_mlp=_B)
	A.applied_token_merged_ratio=B