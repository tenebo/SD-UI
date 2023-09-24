_C='Automatic'
_B=False
_A=None
import os,collections
from dataclasses import dataclass
from modules import paths,shared,devices,script_callbacks,sd_models,extra_networks,lowvram,sd_hijack,hashes
import glob
from copy import deepcopy
vae_path=os.path.abspath(os.path.join(paths.models_path,'VAE'))
vae_ignore_keys={'model_ema.decay','model_ema.num_updates'}
vae_dict={}
base_vae=_A
loaded_vae_file=_A
checkpoint_info=_A
checkpoints_loaded=collections.OrderedDict()
def get_loaded_vae_name():
	if loaded_vae_file is _A:return
	return os.path.basename(loaded_vae_file)
def get_loaded_vae_hash():
	if loaded_vae_file is _A:return
	A=hashes.sha256(loaded_vae_file,'vae');return A[0:10]if A else _A
def get_base_vae(model):
	A=model
	if base_vae is not _A and checkpoint_info==A.sd_checkpoint_info and A:return base_vae
def store_base_vae(model):
	A=model;global base_vae,checkpoint_info
	if checkpoint_info!=A.sd_checkpoint_info:assert not loaded_vae_file,'Trying to store non-base VAE!';base_vae=deepcopy(A.first_stage_model.state_dict());checkpoint_info=A.sd_checkpoint_info
def delete_base_vae():global base_vae,checkpoint_info;base_vae=_A;checkpoint_info=_A
def restore_base_vae(model):
	A=model;global loaded_vae_file
	if base_vae is not _A and checkpoint_info==A.sd_checkpoint_info:print('Restoring base VAE');_load_vae_dict(A,base_vae);loaded_vae_file=_A
	delete_base_vae()
def get_filename(filepath):return os.path.basename(filepath)
def refresh_vae_list():
	B='**/*.safetensors';C='**/*.pt';D='**/*.ckpt';E='**/*.vae.safetensors';F='**/*.vae.pt';G='**/*.vae.ckpt';vae_dict.clear();A=[os.path.join(sd_models.model_path,G),os.path.join(sd_models.model_path,F),os.path.join(sd_models.model_path,E),os.path.join(vae_path,D),os.path.join(vae_path,C),os.path.join(vae_path,B)]
	if shared.cmd_opts.ckpt_dir is not _A and os.path.isdir(shared.cmd_opts.ckpt_dir):A+=[os.path.join(shared.cmd_opts.ckpt_dir,G),os.path.join(shared.cmd_opts.ckpt_dir,F),os.path.join(shared.cmd_opts.ckpt_dir,E)]
	if shared.cmd_opts.vae_dir is not _A and os.path.isdir(shared.cmd_opts.vae_dir):A+=[os.path.join(shared.cmd_opts.vae_dir,D),os.path.join(shared.cmd_opts.vae_dir,C),os.path.join(shared.cmd_opts.vae_dir,B)]
	H=[]
	for J in A:H+=glob.iglob(J,recursive=True)
	for I in H:K=get_filename(I);vae_dict[K]=I
	vae_dict.update(dict(sorted(vae_dict.items(),key=lambda item:shared.natural_sort_key(item[0]))))
def find_vae_near_checkpoint(checkpoint_file):
	B=os.path.basename(checkpoint_file).rsplit('.',1)[0]
	for A in vae_dict.values():
		if os.path.basename(A).startswith(B):return A
@dataclass
class VaeResolution:
	vae:str=_A;source:str=_A;resolved:bool=True
	def tuple(A):return A.vae,A.source
def is_automatic():return shared.opts.sd_vae in{_C,'auto'}
def resolve_vae_from_setting():
	if shared.opts.sd_vae=='None':return VaeResolution()
	A=vae_dict.get(shared.opts.sd_vae,_A)
	if A is not _A:return VaeResolution(A,'specified in settings')
	if not is_automatic():print(f"Couldn't find VAE named {shared.opts.sd_vae}; using None instead")
	return VaeResolution(resolved=_B)
def resolve_vae_from_user_metadata(checkpoint_file):
	C=extra_networks.get_user_metadata(checkpoint_file);A=C.get('vae',_A)
	if A is not _A and A!=_C:
		if A=='None':return VaeResolution()
		B=vae_dict.get(A,_A)
		if B is not _A:return VaeResolution(B,'from user metadata')
	return VaeResolution(resolved=_B)
def resolve_vae_near_checkpoint(checkpoint_file):
	A=find_vae_near_checkpoint(checkpoint_file)
	if A is not _A and(not shared.opts.sd_vae_overrides_per_model_preferences or is_automatic()):return VaeResolution(A,'found near the checkpoint')
	return VaeResolution(resolved=_B)
def resolve_vae(checkpoint_file):
	B=checkpoint_file
	if shared.cmd_opts.vae_path is not _A:return VaeResolution(shared.cmd_opts.vae_path,'from commandline argument')
	if shared.opts.sd_vae_overrides_per_model_preferences and not is_automatic():return resolve_vae_from_setting()
	A=resolve_vae_from_user_metadata(B)
	if A.resolved:return A
	A=resolve_vae_near_checkpoint(B)
	if A.resolved:return A
	A=resolve_vae_from_setting();return A
def load_vae_dict(filename,map_location):A=sd_models.read_state_dict(filename,map_location=map_location);B={A:B for(A,B)in A.items()if A[0:4]!='loss'and A not in vae_ignore_keys};return B
def load_vae(model,vae_file=_A,vae_source='from unknown source'):
	C=vae_source;B=model;A=vae_file;global vae_dict,base_vae,loaded_vae_file;D=shared.opts.sd_vae_checkpoint_cache>0
	if A:
		if D and A in checkpoints_loaded:print(f"Loading VAE weights {C}: cached {get_filename(A)}");store_base_vae(B);_load_vae_dict(B,checkpoints_loaded[A])
		else:
			assert os.path.isfile(A),f"VAE {C} doesn't exist: {A}";print(f"Loading VAE weights {C}: {A}");store_base_vae(B);E=load_vae_dict(A,map_location=shared.weight_load_location);_load_vae_dict(B,E)
			if D:checkpoints_loaded[A]=E.copy()
		if D:
			while len(checkpoints_loaded)>shared.opts.sd_vae_checkpoint_cache+1:checkpoints_loaded.popitem(last=_B)
		F=get_filename(A)
		if F not in vae_dict:vae_dict[F]=A
	elif loaded_vae_file:restore_base_vae(B)
	loaded_vae_file=A;B.base_vae=base_vae;B.loaded_vae_file=loaded_vae_file
def _load_vae_dict(model,vae_dict_1):A=model;A.first_stage_model.load_state_dict(vae_dict_1);A.first_stage_model.to(devices.dtype_vae)
def clear_loaded_vae():global loaded_vae_file;loaded_vae_file=_A
unspecified=object()
def reload_vae_weights(sd_model=_A,vae_file=unspecified):
	B=vae_file;A=sd_model
	if not A:A=shared.sd_model
	D=A.sd_checkpoint_info;E=D.filename
	if B==unspecified:B,C=resolve_vae(E).tuple()
	else:C='from function argument'
	if loaded_vae_file==B:return
	if A.lowvram:lowvram.send_everything_to_cpu()
	else:A.to(devices.cpu)
	sd_hijack.model_hijack.undo_hijack(A);load_vae(A,B,C);sd_hijack.model_hijack.hijack(A);script_callbacks.model_loaded_callback(A)
	if not A.lowvram:A.to(devices.device)
	print('VAE weights loaded.');return A