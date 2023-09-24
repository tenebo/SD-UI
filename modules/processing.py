from __future__ import annotations
_P='VAE Encoder'
_O='clip_skip'
_N='image_cfg_scale'
_M='crossattn-adm'
_L='Full'
_K='denoising_strength'
_J='inpainting_mask_weight'
_I='concat'
_H='hybrid'
_G='RGBa'
_F='RGB'
_E='RGBA'
_D=1.
_C=True
_B=False
_A=None
import json,logging,math,os,sys,hashlib
from dataclasses import dataclass,field
import torch,numpy as np
from PIL import Image,ImageOps
import random,cv2
from skimage import exposure
from typing import Any
import modules.sd_hijack
from modules import devices,prompt_parser,masking,sd_samplers,lowvram,generation_parameters_copypaste,extra_networks,sd_vae_approx,scripts,sd_samplers_common,sd_unet,errors,rng
from modules.rng import slerp
from modules.sd_hijack import model_hijack
from modules.sd_samplers_common import images_tensor_to_samples,decode_first_stage,approximation_indexes
from modules.shared import opts,cmd_opts,state
import modules.shared as shared,modules.paths as paths,modules.face_restoration,modules.images as images,modules.styles,modules.sd_models as sd_models,modules.sd_vae as sd_vae
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
from einops import repeat,rearrange
from blendmodes.blend import blendLayers,BlendType
opt_C=4
opt_f=8
def setup_color_correction(image):logging.info('Calibrating color correction.');A=cv2.cvtColor(np.asarray(image.copy()),cv2.COLOR_RGB2LAB);return A
def apply_color_correction(correction,original_image):B=original_image;logging.info('Applying color correction.');A=Image.fromarray(cv2.cvtColor(exposure.match_histograms(cv2.cvtColor(np.asarray(B),cv2.COLOR_RGB2LAB),correction,channel_axis=2),cv2.COLOR_LAB2RGB).astype('uint8'));A=blendLayers(A,B,BlendType.LUMINOSITY);return A.convert(_F)
def apply_overlay(image,paste_loc,index,overlays):
	E=index;D=paste_loc;B=overlays;A=image
	if B is _A or E>=len(B):return A
	C=B[E]
	if D is not _A:G,H,I,J=D;F=Image.new(_E,(C.width,C.height));A=images.resize_image(1,A,I,J);F.paste(A,(G,H));A=F
	A=A.convert(_E);A.alpha_composite(C);A=A.convert(_F);return A
def create_binary_mask(image):
	A=image
	if A.mode==_E and A.getextrema()[-1]!=(255,255):A=A.split()[-1].convert('L').point(lambda x:255 if x>128 else 0)
	else:A=A.convert('L')
	return A
def txt2img_image_conditioning(sd_model,x,width,height):
	B=sd_model
	if B.model.conditioning_key in{_H,_I}:A=torch.ones(x.shape[0],3,height,width,device=x.device)*.5;A=images_tensor_to_samples(A,approximation_indexes.get(opts.sd_vae_encode_method));A=torch.nn.functional.pad(A,(0,0,0,0,1,0),value=_D);A=A.to(x.dtype);return A
	elif B.model.conditioning_key==_M:return x.new_zeros(x.shape[0],2*B.noise_augmentor.time_embed.dim,dtype=x.dtype,device=x.device)
	else:return x.new_zeros(x.shape[0],5,1,1,dtype=x.dtype,device=x.device)
@dataclass(repr=_B)
class StandardDemoProcessing:
	sd_model:object=_A;outpath_samples:str=_A;outpath_grids:str=_A;prompt:str='';prompt_for_display:str=_A;negative_prompt:str='';styles:list[str]=_A;seed:int=-1;subseed:int=-1;subseed_strength:float=0;seed_resize_from_h:int=-1;seed_resize_from_w:int=-1;seed_enable_extras:bool=_C;sampler_name:str=_A;batch_size:int=1;n_iter:int=1;steps:int=50;cfg_scale:float=7.;width:int=512;height:int=512;restore_faces:bool=_A;tiling:bool=_A;do_not_save_samples:bool=_B;do_not_save_grid:bool=_B;extra_generation_params:dict[str,Any]=_A;overlay_images:list=_A;eta:float=_A;do_not_reload_embeddings:bool=_B;denoising_strength:float=0;ddim_discretize:str=_A;s_min_uncond:float=_A;s_churn:float=_A;s_tmax:float=_A;s_tmin:float=_A;s_noise:float=_A;override_settings:dict[str,Any]=_A;override_settings_restore_afterwards:bool=_C;sampler_index:int=_A;refiner_checkpoint:str=_A;refiner_switch_at:float=_A;token_merging_ratio=0;token_merging_ratio_hr=0;disable_extra_networks:bool=_B;scripts_value:scripts.ScriptRunner=field(default=_A,init=_B);script_args_value:list=field(default=_A,init=_B);scripts_setup_complete:bool=field(default=_B,init=_B);cached_uc=[_A,_A];cached_c=[_A,_A];comments:dict=_A;sampler:sd_samplers_common.Sampler|_A=field(default=_A,init=_B);is_using_inpainting_conditioning:bool=field(default=_B,init=_B);paste_to:tuple|_A=field(default=_A,init=_B);is_hr_pass:bool=field(default=_B,init=_B);c:tuple=field(default=_A,init=_B);uc:tuple=field(default=_A,init=_B);rng:rng.ImageRNG|_A=field(default=_A,init=_B);step_multiplier:int=field(default=1,init=_B);color_corrections:list=field(default=_A,init=_B);all_prompts:list=field(default=_A,init=_B);all_negative_prompts:list=field(default=_A,init=_B);all_seeds:list=field(default=_A,init=_B);all_subseeds:list=field(default=_A,init=_B);iteration:int=field(default=0,init=_B);main_prompt:str=field(default=_A,init=_B);main_negative_prompt:str=field(default=_A,init=_B);prompts:list=field(default=_A,init=_B);negative_prompts:list=field(default=_A,init=_B);seeds:list=field(default=_A,init=_B);subseeds:list=field(default=_A,init=_B);extra_network_data:dict=field(default=_A,init=_B);user:str=field(default=_A,init=_B);sd_model_name:str=field(default=_A,init=_B);sd_model_hash:str=field(default=_A,init=_B);sd_vae_name:str=field(default=_A,init=_B);sd_vae_hash:str=field(default=_A,init=_B);is_api:bool=field(default=_B,init=_B)
	def __post_init__(A):
		if A.sampler_index is not _A:print('sampler_index argument for StandardDemoProcessing does not do anything; use sampler_name',file=sys.stderr)
		A.comments={}
		if A.styles is _A:A.styles=[]
		A.sampler_noise_scheduler_override=_A;A.s_min_uncond=A.s_min_uncond if A.s_min_uncond is not _A else opts.s_min_uncond;A.s_churn=A.s_churn if A.s_churn is not _A else opts.s_churn;A.s_tmin=A.s_tmin if A.s_tmin is not _A else opts.s_tmin;A.s_tmax=(A.s_tmax if A.s_tmax is not _A else opts.s_tmax)or float('inf');A.s_noise=A.s_noise if A.s_noise is not _A else opts.s_noise;A.extra_generation_params=A.extra_generation_params or{};A.override_settings=A.override_settings or{};A.script_args=A.script_args or{};A.refiner_checkpoint_info=_A
		if not A.seed_enable_extras:A.subseed=-1;A.subseed_strength=0;A.seed_resize_from_h=0;A.seed_resize_from_w=0
		A.cached_uc=StandardDemoProcessing.cached_uc;A.cached_c=StandardDemoProcessing.cached_c
	@property
	def sd_model(self):return shared.sd_model
	@sd_model.setter
	def sd_model(self,value):0
	@property
	def scripts(self):return self.scripts_value
	@scripts.setter
	def scripts(self,value):
		A=self;A.scripts_value=value
		if A.scripts_value and A.script_args_value and not A.scripts_setup_complete:A.setup_scripts()
	@property
	def script_args(self):return self.script_args_value
	@script_args.setter
	def script_args(self,value):
		A=self;A.script_args_value=value
		if A.scripts_value and A.script_args_value and not A.scripts_setup_complete:A.setup_scripts()
	def setup_scripts(A):A.scripts_setup_complete=_C;A.scripts.setup_scrips(A,is_ui=not A.is_api)
	def comment(A,text):A.comments[text]=1
	def txt2img_image_conditioning(A,x,width=_A,height=_A):A.is_using_inpainting_conditioning=A.sd_model.model.conditioning_key in{_H,_I};return txt2img_image_conditioning(A.sd_model,x,width or A.width,height or A.height)
	def depth2img_image_conditioning(C,source_image):D=source_image;F=AddMiDaS(model_type='dpt_hybrid');G=F({'jpg':rearrange(D[0],'c h w -> h w c')});B=torch.from_numpy(G['midas_in'][_A,...]).to(device=shared.device);B=repeat(B,'1 ... -> n ...',n=C.batch_size);H=images_tensor_to_samples(D*.5+.5,approximation_indexes.get(opts.sd_vae_encode_method));A=torch.nn.functional.interpolate(C.sd_model.depth_model(B),size=H.shape[2:],mode='bicubic',align_corners=_B);E,I=torch.aminmax(A);A=2.*(A-E)/(I-E)-_D;return A
	def edit_image_conditioning(B,source_image):A=images_tensor_to_samples(source_image*.5+.5,approximation_indexes.get(opts.sd_vae_encode_method));return A
	def unclip_image_conditioning(B,source_image):
		A=B.sd_model.embedder(source_image)
		if B.sd_model.noise_augmentor is not _A:C=0;A,D=B.sd_model.noise_augmentor(A,noise_level=repeat(torch.tensor([C]).to(A.device),'1 -> b',b=A.shape[0]));A=torch.cat((A,D),1)
		return A
	def inpainting_image_conditioning(C,source_image,latent_image,image_mask=_A):
		D=image_mask;B=source_image;C.is_using_inpainting_conditioning=_C
		if D is not _A:
			if torch.is_tensor(D):A=D
			else:A=np.array(D.convert('L'));A=A.astype(np.float32)/255.;A=torch.from_numpy(A[_A,_A]);A=torch.round(A)
		else:A=B.new_ones(1,1,*B.shape[-2:])
		A=A.to(device=B.device,dtype=B.dtype);E=torch.lerp(B,B*(_D-A),getattr(C,_J,shared.opts.inpainting_mask_weight));E=C.sd_model.get_first_stage_encoding(C.sd_model.encode_first_stage(E));A=torch.nn.functional.interpolate(A,size=latent_image.shape[-2:]);A=A.expand(E.shape[0],-1,-1,-1);F=torch.cat([A,E],dim=1);F=F.to(shared.device).type(C.sd_model.dtype);return F
	def img2img_image_conditioning(A,source_image,latent_image,image_mask=_A):
		C=latent_image;B=source_image;B=devices.cond_cast_float(B)
		if isinstance(A.sd_model,LatentDepth2ImageDiffusion):return A.depth2img_image_conditioning(B)
		if A.sd_model.cond_stage_key=='edit':return A.edit_image_conditioning(B)
		if A.sampler.conditioning_key in{_H,_I}:return A.inpainting_image_conditioning(B,C,image_mask=image_mask)
		if A.sampler.conditioning_key==_M:return A.unclip_image_conditioning(B)
		return C.new_zeros(C.shape[0],5,1,1)
	def init(A,all_prompts,all_seeds,all_subseeds):0
	def sample(A,conditioning,unconditional_conditioning,seeds,subseeds,subseed_strength,prompts):raise NotImplementedError()
	def close(A):
		A.sampler=_A;A.c=_A;A.uc=_A
		if not opts.persistent_cond_cache:StandardDemoProcessing.cached_c=[_A,_A];StandardDemoProcessing.cached_uc=[_A,_A]
	def get_token_merging_ratio(A,for_hr=_B):
		if for_hr:return A.token_merging_ratio_hr or opts.token_merging_ratio_hr or A.token_merging_ratio or opts.token_merging_ratio
		return A.token_merging_ratio or opts.token_merging_ratio
	def setup_prompts(A):
		if isinstance(A.prompt,list):A.all_prompts=A.prompt
		elif isinstance(A.negative_prompt,list):A.all_prompts=[A.prompt]*len(A.negative_prompt)
		else:A.all_prompts=A.batch_size*A.n_iter*[A.prompt]
		if isinstance(A.negative_prompt,list):A.all_negative_prompts=A.negative_prompt
		else:A.all_negative_prompts=[A.negative_prompt]*len(A.all_prompts)
		if len(A.all_prompts)!=len(A.all_negative_prompts):raise RuntimeError(f"Received a different number of prompts ({len(A.all_prompts)}) and negative prompts ({len(A.all_negative_prompts)})")
		A.all_prompts=[shared.prompt_styles.apply_styles_to_prompt(B,A.styles)for B in A.all_prompts];A.all_negative_prompts=[shared.prompt_styles.apply_negative_styles_to_prompt(B,A.styles)for B in A.all_negative_prompts];A.main_prompt=A.all_prompts[0];A.main_negative_prompt=A.all_negative_prompts[0]
	def cached_params(A,required_prompts,steps,extra_network_data,hires_steps=_A,use_old_scheduling=_B):'Returns parameters that invalidate the cond cache if changed';return required_prompts,steps,hires_steps,use_old_scheduling,opts.CLIP_stop_at_last_layers,shared.sd_model.sd_checkpoint_info,extra_network_data,opts.sdxl_crop_left,opts.sdxl_crop_top,A.width,A.height
	def get_conds_with_caching(E,function,required_prompts,steps,caches,extra_network_data,hires_steps=_A):
		'\n        Returns the result of calling function(shared.sd_model, required_prompts, steps)\n        using a cache to store the result if the same arguments have been used before.\n\n        cache is an array containing two elements. The first element is a tuple\n        representing the previously used arguments, or None if no arguments\n        have been used before. The second element is where the previously\n        computed result is stored.\n\n        caches is a list with items described above.\n        ';F=caches;D=hires_steps;C=steps;B=required_prompts
		if shared.opts.use_old_scheduling:
			H=prompt_parser.get_learned_conditioning_prompt_schedules(B,C,D,_B);I=prompt_parser.get_learned_conditioning_prompt_schedules(B,C,D,_C)
			if H!=I:E.extra_generation_params['Old prompt editing timelines']=_C
		G=E.cached_params(B,C,extra_network_data,D,shared.opts.use_old_scheduling)
		for A in F:
			if A[0]is not _A and G==A[0]:return A[1]
		A=F[0]
		with devices.autocast():A[1]=function(shared.sd_model,B,C,D,shared.opts.use_old_scheduling)
		A[0]=G;return A[1]
	def setup_conds(A):D=prompt_parser.SdConditioning(A.prompts,width=A.width,height=A.height);E=prompt_parser.SdConditioning(A.negative_prompts,width=A.width,height=A.height,is_negative_prompt=_C);C=sd_samplers.find_sampler_config(A.sampler_name);B=C.total_steps(A.steps)if C else A.steps;A.step_multiplier=B//A.steps;A.firstpass_steps=B;A.uc=A.get_conds_with_caching(prompt_parser.get_learned_conditioning,E,B,[A.cached_uc],A.extra_network_data);A.c=A.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning,D,B,[A.cached_c],A.extra_network_data)
	def get_conds(A):return A.c,A.uc
	def parse_extra_network_prompts(A):A.prompts,A.extra_network_data=extra_networks.parse_prompts(A.prompts)
	def save_samples(A):'Returns whether generated images need to be written to disk';return opts.samples_save and not A.do_not_save_samples and(opts.save_incomplete_images or not state.interrupted and not state.skipped)
class Processed:
	def __init__(A,p,images_list,seed=-1,info='',subseed=_A,all_prompts=_A,all_negative_prompts=_A,all_seeds=_A,all_subseeds=_A,index_of_first_image=0,infotexts=_A,comments=''):A.images=images_list;A.prompt=p.prompt;A.negative_prompt=p.negative_prompt;A.seed=seed;A.subseed=subseed;A.subseed_strength=p.subseed_strength;A.info=info;A.comments=''.join(f"{A}\n"for A in p.comments);A.width=p.width;A.height=p.height;A.sampler_name=p.sampler_name;A.cfg_scale=p.cfg_scale;A.image_cfg_scale=getattr(p,_N,_A);A.steps=p.steps;A.batch_size=p.batch_size;A.restore_faces=p.restore_faces;A.face_restoration_model=opts.face_restoration_model if p.restore_faces else _A;A.sd_model_name=p.sd_model_name;A.sd_model_hash=p.sd_model_hash;A.sd_vae_name=p.sd_vae_name;A.sd_vae_hash=p.sd_vae_hash;A.seed_resize_from_w=p.seed_resize_from_w;A.seed_resize_from_h=p.seed_resize_from_h;A.denoising_strength=getattr(p,_K,_A);A.extra_generation_params=p.extra_generation_params;A.index_of_first_image=index_of_first_image;A.styles=p.styles;A.job_timestamp=state.job_timestamp;A.clip_skip=opts.CLIP_stop_at_last_layers;A.token_merging_ratio=p.token_merging_ratio;A.token_merging_ratio_hr=p.token_merging_ratio_hr;A.eta=p.eta;A.ddim_discretize=p.ddim_discretize;A.s_churn=p.s_churn;A.s_tmin=p.s_tmin;A.s_tmax=p.s_tmax;A.s_noise=p.s_noise;A.s_min_uncond=p.s_min_uncond;A.sampler_noise_scheduler_override=p.sampler_noise_scheduler_override;A.prompt=A.prompt if not isinstance(A.prompt,list)else A.prompt[0];A.negative_prompt=A.negative_prompt if not isinstance(A.negative_prompt,list)else A.negative_prompt[0];A.seed=int(A.seed if not isinstance(A.seed,list)else A.seed[0])if A.seed is not _A else-1;A.subseed=int(A.subseed if not isinstance(A.subseed,list)else A.subseed[0])if A.subseed is not _A else-1;A.is_using_inpainting_conditioning=p.is_using_inpainting_conditioning;A.all_prompts=all_prompts or p.all_prompts or[A.prompt];A.all_negative_prompts=all_negative_prompts or p.all_negative_prompts or[A.negative_prompt];A.all_seeds=all_seeds or p.all_seeds or[A.seed];A.all_subseeds=all_subseeds or p.all_subseeds or[A.subseed];A.infotexts=infotexts or[info]
	def js(A):B={'prompt':A.all_prompts[0],'all_prompts':A.all_prompts,'negative_prompt':A.all_negative_prompts[0],'all_negative_prompts':A.all_negative_prompts,'seed':A.seed,'all_seeds':A.all_seeds,'subseed':A.subseed,'all_subseeds':A.all_subseeds,'subseed_strength':A.subseed_strength,'width':A.width,'height':A.height,'sampler_name':A.sampler_name,'cfg_scale':A.cfg_scale,'steps':A.steps,'batch_size':A.batch_size,'restore_faces':A.restore_faces,'face_restoration_model':A.face_restoration_model,'sd_model_name':A.sd_model_name,'sd_model_hash':A.sd_model_hash,'sd_vae_name':A.sd_vae_name,'sd_vae_hash':A.sd_vae_hash,'seed_resize_from_w':A.seed_resize_from_w,'seed_resize_from_h':A.seed_resize_from_h,_K:A.denoising_strength,'extra_generation_params':A.extra_generation_params,'index_of_first_image':A.index_of_first_image,'infotexts':A.infotexts,'styles':A.styles,'job_timestamp':A.job_timestamp,_O:A.clip_skip,'is_using_inpainting_conditioning':A.is_using_inpainting_conditioning};return json.dumps(B)
	def infotext(A,p,index):B=index;return create_infotext(p,A.all_prompts,A.all_seeds,A.all_subseeds,comments=[],position_in_batch=B%A.batch_size,iteration=B//A.batch_size)
	def get_token_merging_ratio(A,for_hr=_B):return A.token_merging_ratio_hr if for_hr else A.token_merging_ratio
def create_random_tensors(shape,seeds,subseeds=_A,subseed_strength=.0,seed_resize_from_h=0,seed_resize_from_w=0,p=_A):A=rng.ImageRNG(shape,seeds,subseeds=subseeds,subseed_strength=subseed_strength,seed_resize_from_h=seed_resize_from_h,seed_resize_from_w=seed_resize_from_w);return A.next()
class DecodedSamples(list):already_decoded=_C
def decode_latent_batch(model,batch,target_device=_A,check_for_nans=_B):
	E=target_device;D=model;A=batch;F=DecodedSamples()
	for C in range(A.shape[0]):
		B=decode_first_stage(D,A[C:C+1])[0]
		if check_for_nans:
			try:devices.test_for_nans(B,'vae')
			except devices.NansException as G:
				if devices.dtype_vae==torch.float32 or not shared.opts.auto_vae_precision:raise G
				errors.print_error_explanation("A tensor with all NaNs was produced in VAE.\nWeb UI will now convert VAE into 32-bit float and retry.\nTo disable this behavior, disable the 'Automatically revert VAE to 32-bit floats' setting.\nTo always start with 32-bit VAE, use --no-half-vae commandline flag.");devices.dtype_vae=torch.float32;D.first_stage_model.to(devices.dtype_vae);A=A.to(devices.dtype_vae);B=decode_first_stage(D,A[C:C+1])[0]
		if E is not _A:B=B.to(E)
		F.append(B)
	return F
def get_fixed_seed(seed):
	A=seed
	if A==''or A is _A:A=-1
	elif isinstance(A,str):
		try:A=int(A)
		except Exception:A=-1
	if A==-1:return int(random.randrange(4294967294))
	return A
def fix_seed(p):p.seed=get_fixed_seed(p.seed);p.subseed=get_fixed_seed(p.subseed)
def program_version():
	import launch as B;A=B.git_tag()
	if A=='<none>':A=_A
	return A
def create_infotext(p,all_prompts,all_seeds,all_subseeds,comments=_A,iteration=0,position_in_batch=0,use_main_prompt=_B,index=_A,all_negative_prompts=_A):
	C=all_negative_prompts;B=use_main_prompt;A=index
	if A is _A:A=position_in_batch+iteration*p.batch_size
	if C is _A:C=p.all_negative_prompts
	E=getattr(p,_O,opts.CLIP_stop_at_last_layers);H=getattr(p,'enable_hr',_B);F=p.get_token_merging_ratio();G=p.get_token_merging_ratio(for_hr=_C);D=opts.eta_noise_seed_delta!=0
	if D:D=sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p)
	I={'Steps':p.steps,'Sampler':p.sampler_name,'CFG scale':p.cfg_scale,'Image CFG scale':getattr(p,_N,_A),'Seed':p.all_seeds[0]if B else all_seeds[A],'Face restoration':opts.face_restoration_model if p.restore_faces else _A,'Size':f"{p.width}x{p.height}",'Model hash':p.sd_model_hash if opts.add_model_hash_to_info else _A,'Model':p.sd_model_name if opts.add_model_name_to_info else _A,'VAE hash':p.sd_vae_hash if opts.add_model_hash_to_info else _A,'VAE':p.sd_vae_name if opts.add_model_name_to_info else _A,'Variation seed':_A if p.subseed_strength==0 else p.all_subseeds[0]if B else all_subseeds[A],'Variation seed strength':_A if p.subseed_strength==0 else p.subseed_strength,'Seed resize from':_A if p.seed_resize_from_w<=0 or p.seed_resize_from_h<=0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}",'Denoising strength':getattr(p,_K,_A),'Conditional mask weight':getattr(p,_J,shared.opts.inpainting_mask_weight)if p.is_using_inpainting_conditioning else _A,'Clip skip':_A if E<=1 else E,'ENSD':opts.eta_noise_seed_delta if D else _A,'Token merging ratio':_A if F==0 else F,'Token merging ratio hr':_A if not H or G==0 else G,'Init image hash':getattr(p,'init_img_hash',_A),'RNG':opts.randn_source if opts.randn_source!='GPU'else _A,'NGMS':_A if p.s_min_uncond==0 else p.s_min_uncond,'Tiling':'True'if p.tiling else _A,**p.extra_generation_params,'Version':program_version()if opts.add_version_to_infotext else _A,'User':p.user if opts.add_user_name_to_info else _A};J=', '.join([A if A==B else f"{A}: {generation_parameters_copypaste.quote(B)}"for(A,B)in I.items()if B is not _A]);K=p.main_prompt if B else all_prompts[A];L=f"\nNegative prompt: {p.main_negative_prompt if B else C[A]}"if C[A]else'';return f"{K}{L}\n{J}".strip()
def process_images(p):
	D='sd_vae';C='sd_model_checkpoint'
	if p.scripts is not _A:p.scripts.before_process(p)
	E={A:opts.data[A]for A in p.override_settings.keys()}
	try:
		if sd_models.checkpoint_aliases.get(p.override_settings.get(C))is _A:p.override_settings.pop(C,_A);sd_models.reload_model_weights()
		for(A,B)in p.override_settings.items():
			opts.set(A,B,is_api=_C,run_callbacks=_B)
			if A==C:sd_models.reload_model_weights()
			if A==D:sd_vae.reload_vae_weights()
		sd_models.apply_token_merging(p.sd_model,p.get_token_merging_ratio());F=process_images_inner(p)
	finally:
		sd_models.apply_token_merging(p.sd_model,0)
		if p.override_settings_restore_afterwards:
			for(A,B)in E.items():
				setattr(opts,A,B)
				if A==D:sd_vae.reload_vae_weights()
	return F
def process_images_inner(p):
	'this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch';U='parameters'
	if isinstance(p.prompt,list):assert len(p.prompt)>0
	else:assert p.prompt is not _A
	devices.torch_gc();L=get_fixed_seed(p.seed);M=get_fixed_seed(p.subseed)
	if p.restore_faces is _A:p.restore_faces=opts.face_restoration
	if p.tiling is _A:p.tiling=opts.tiling
	if p.refiner_checkpoint not in(_A,'','None','none'):
		p.refiner_checkpoint_info=sd_models.get_closet_checkpoint_match(p.refiner_checkpoint)
		if p.refiner_checkpoint_info is _A:raise Exception(f"Could not find checkpoint with name {p.refiner_checkpoint}")
	p.sd_model_name=shared.sd_model.sd_checkpoint_info.name_for_extra;p.sd_model_hash=shared.sd_model.sd_model_hash;p.sd_vae_name=sd_vae.get_loaded_vae_name();p.sd_vae_hash=sd_vae.get_loaded_vae_hash();modules.sd_hijack.model_hijack.apply_circular(p.tiling);modules.sd_hijack.model_hijack.clear_comments();p.setup_prompts()
	if isinstance(L,list):p.all_seeds=L
	else:p.all_seeds=[int(L)+(A if p.subseed_strength==0 else 0)for A in range(len(p.all_prompts))]
	if isinstance(M,list):p.all_subseeds=M
	else:p.all_subseeds=[int(M)+A for A in range(len(p.all_prompts))]
	if os.path.exists(cmd_opts.embeddings_dir)and not p.do_not_reload_embeddings:model_hijack.embedding_db.load_textual_inversion_embeddings()
	if p.scripts is not _A:p.scripts.process(p)
	I=[];G=[]
	with torch.no_grad(),p.sd_model.ema_scope():
		with devices.autocast():
			p.init(p.all_prompts,p.all_seeds,p.all_subseeds)
			if shared.opts.live_previews_enable and opts.show_progress_type=='Approx NN':sd_vae_approx.model()
			sd_unet.apply_unet()
		if state.job_count==-1:state.job_count=p.n_iter
		for B in range(p.n_iter):
			p.iteration=B
			if state.skipped:state.skipped=_B
			if state.interrupted:break
			sd_models.reload_model_weights();p.prompts=p.all_prompts[B*p.batch_size:(B+1)*p.batch_size];p.negative_prompts=p.all_negative_prompts[B*p.batch_size:(B+1)*p.batch_size];p.seeds=p.all_seeds[B*p.batch_size:(B+1)*p.batch_size];p.subseeds=p.all_subseeds[B*p.batch_size:(B+1)*p.batch_size];p.rng=rng.ImageRNG((opt_C,p.height//opt_f,p.width//opt_f),p.seeds,subseeds=p.subseeds,subseed_strength=p.subseed_strength,seed_resize_from_h=p.seed_resize_from_h,seed_resize_from_w=p.seed_resize_from_w)
			if p.scripts is not _A:p.scripts.before_process_batch(p,batch_number=B,prompts=p.prompts,seeds=p.seeds,subseeds=p.subseeds)
			if len(p.prompts)==0:break
			p.parse_extra_network_prompts()
			if not p.disable_extra_networks:
				with devices.autocast():extra_networks.activate(p,p.extra_network_data)
			if p.scripts is not _A:p.scripts.process_batch(p,batch_number=B,prompts=p.prompts,seeds=p.seeds,subseeds=p.subseeds)
			if B==0:
				with open(os.path.join(paths.data_path,'params.txt'),'w',encoding='utf8')as V:W=Processed(p,[]);V.write(W.infotext(p,0))
			p.setup_conds()
			for X in model_hijack.comments:p.comment(X)
			p.extra_generation_params.update(model_hijack.extra_generation_params)
			if p.n_iter>1:shared.state.job=f"Batch {B+1} out of {p.n_iter}"
			with devices.without_autocast()if devices.unet_needs_upcast else devices.autocast():J=p.sample(conditioning=p.c,unconditional_conditioning=p.uc,seeds=p.seeds,subseeds=p.subseeds,subseed_strength=p.subseed_strength,prompts=p.prompts)
			if getattr(J,'already_decoded',_B):D=J
			else:
				if opts.sd_vae_decode_method!=_L:p.extra_generation_params['VAE Decoder']=opts.sd_vae_decode_method
				D=decode_latent_batch(p.sd_model,J,target_device=devices.cpu,check_for_nans=_C)
			D=torch.stack(D).float();D=torch.clamp((D+_D)/2.,min=.0,max=_D);del J
			if lowvram.is_enabled(shared.sd_model):lowvram.send_everything_to_cpu()
			devices.torch_gc()
			if p.scripts is not _A:p.scripts.postprocess_batch(p,D,batch_number=B);p.prompts=p.all_prompts[B*p.batch_size:(B+1)*p.batch_size];p.negative_prompts=p.all_negative_prompts[B*p.batch_size:(B+1)*p.batch_size];O=scripts.PostprocessBatchListArgs(list(D));p.scripts.postprocess_batch_list(p,O,batch_number=B);D=O.images
			def E(index=0,use_main_prompt=_B):return create_infotext(p,p.prompts,p.seeds,p.subseeds,use_main_prompt=use_main_prompt,index=index,all_negative_prompts=p.negative_prompts)
			K=p.save_samples()
			for(A,F)in enumerate(D):
				p.batch_index=A;F=255.*np.moveaxis(F.cpu().numpy(),0,2);F=F.astype(np.uint8)
				if p.restore_faces:
					if K and opts.save_images_before_face_restoration:images.save_image(Image.fromarray(F),p.outpath_samples,'',p.seeds[A],p.prompts[A],opts.samples_format,info=E(A),p=p,suffix='-before-face-restoration')
					devices.torch_gc();F=modules.face_restoration.restore_faces(F);devices.torch_gc()
				C=Image.fromarray(F)
				if p.scripts is not _A:P=scripts.PostprocessImageArgs(C);p.scripts.postprocess_image(p,P);C=P.image
				if p.color_corrections is not _A and A<len(p.color_corrections):
					if K and opts.save_images_before_color_correction:Y=apply_overlay(C,p.paste_to,A,p.overlay_images);images.save_image(Y,p.outpath_samples,'',p.seeds[A],p.prompts[A],opts.samples_format,info=E(A),p=p,suffix='-before-color-correction')
					C=apply_color_correction(p.color_corrections[A],C)
				C=apply_overlay(C,p.paste_to,A,p.overlay_images)
				if K:images.save_image(C,p.outpath_samples,'',p.seeds[A],p.prompts[A],opts.samples_format,info=E(A),p=p)
				H=E(A);I.append(H)
				if opts.enable_pnginfo:C.info[U]=H
				G.append(C)
				if K and hasattr(p,'mask_for_overlay')and p.mask_for_overlay and any([opts.save_mask,opts.save_mask_composite,opts.return_mask,opts.return_mask_composite]):
					Q=p.mask_for_overlay.convert(_F);R=Image.composite(C.convert(_E).convert(_G),Image.new(_G,C.size),images.resize_image(2,p.mask_for_overlay,C.width,C.height).convert('L')).convert(_E)
					if opts.save_mask:images.save_image(Q,p.outpath_samples,'',p.seeds[A],p.prompts[A],opts.samples_format,info=E(A),p=p,suffix='-mask')
					if opts.save_mask_composite:images.save_image(R,p.outpath_samples,'',p.seeds[A],p.prompts[A],opts.samples_format,info=E(A),p=p,suffix='-mask-composite')
					if opts.return_mask:G.append(Q)
					if opts.return_mask_composite:G.append(R)
			del D;devices.torch_gc();state.nextjob()
		p.color_corrections=_A;S=0;Z=len(G)<2 and opts.grid_only_if_multiple
		if(opts.return_grid or opts.grid_save)and not p.do_not_save_grid and not Z:
			N=images.image_grid(G,p.batch_size)
			if opts.return_grid:
				H=E(use_main_prompt=_C);I.insert(0,H)
				if opts.enable_pnginfo:N.info[U]=H
				G.insert(0,N);S=1
			if opts.grid_save:images.save_image(N,p.outpath_grids,'grid',p.all_seeds[0],p.all_prompts[0],opts.grid_format,info=E(use_main_prompt=_C),short_filename=not opts.grid_extended_filename,p=p,grid=_C)
	if not p.disable_extra_networks and p.extra_network_data:extra_networks.deactivate(p,p.extra_network_data)
	devices.torch_gc();T=Processed(p,images_list=G,seed=p.all_seeds[0],info=I[0],subseed=p.all_subseeds[0],index_of_first_image=S,infotexts=I)
	if p.scripts is not _A:p.scripts.postprocess(p,T)
	return T
def old_hires_fix_first_pass_dimensions(width,height):'old algorithm for auto-calculating first pass size';B=height;A=width;D=512*512;E=A*B;C=math.sqrt(D/E);A=math.ceil(C*A/64)*64;B=math.ceil(C*B/64)*64;return A,B
@dataclass(repr=_B)
class StandardDemoProcessingTxt2Img(StandardDemoProcessing):
	enable_hr:bool=_B;denoising_strength:float=.75;firstphase_width:int=0;firstphase_height:int=0;hr_scale:float=2.;hr_upscaler:str=_A;hr_second_pass_steps:int=0;hr_resize_x:int=0;hr_resize_y:int=0;hr_checkpoint_name:str=_A;hr_sampler_name:str=_A;hr_prompt:str='';hr_negative_prompt:str='';cached_hr_uc=[_A,_A];cached_hr_c=[_A,_A];hr_checkpoint_info:dict=field(default=_A,init=_B);hr_upscale_to_x:int=field(default=0,init=_B);hr_upscale_to_y:int=field(default=0,init=_B);truncate_x:int=field(default=0,init=_B);truncate_y:int=field(default=0,init=_B);applied_old_hires_behavior_to:tuple=field(default=_A,init=_B);latent_scale_mode:dict=field(default=_A,init=_B);hr_c:tuple|_A=field(default=_A,init=_B);hr_uc:tuple|_A=field(default=_A,init=_B);all_hr_prompts:list=field(default=_A,init=_B);all_hr_negative_prompts:list=field(default=_A,init=_B);hr_prompts:list=field(default=_A,init=_B);hr_negative_prompts:list=field(default=_A,init=_B);hr_extra_network_data:list=field(default=_A,init=_B)
	def __post_init__(A):
		super().__post_init__()
		if A.firstphase_width!=0 or A.firstphase_height!=0:A.hr_upscale_to_x=A.width;A.hr_upscale_to_y=A.height;A.width=A.firstphase_width;A.height=A.firstphase_height
		A.cached_hr_uc=StandardDemoProcessingTxt2Img.cached_hr_uc;A.cached_hr_c=StandardDemoProcessingTxt2Img.cached_hr_c
	def calculate_target_resolution(A):
		if opts.use_old_hires_fix_width_height and A.applied_old_hires_behavior_to!=(A.width,A.height):A.hr_resize_x=A.width;A.hr_resize_y=A.height;A.hr_upscale_to_x=A.width;A.hr_upscale_to_y=A.height;A.width,A.height=old_hires_fix_first_pass_dimensions(A.width,A.height);A.applied_old_hires_behavior_to=A.width,A.height
		if A.hr_resize_x==0 and A.hr_resize_y==0:A.extra_generation_params['Hires upscale']=A.hr_scale;A.hr_upscale_to_x=int(A.width*A.hr_scale);A.hr_upscale_to_y=int(A.height*A.hr_scale)
		else:
			A.extra_generation_params['Hires resize']=f"{A.hr_resize_x}x{A.hr_resize_y}"
			if A.hr_resize_y==0:A.hr_upscale_to_x=A.hr_resize_x;A.hr_upscale_to_y=A.hr_resize_x*A.height//A.width
			elif A.hr_resize_x==0:A.hr_upscale_to_x=A.hr_resize_y*A.width//A.height;A.hr_upscale_to_y=A.hr_resize_y
			else:
				B=A.hr_resize_x;C=A.hr_resize_y;D=A.width/A.height;E=A.hr_resize_x/A.hr_resize_y
				if D<E:A.hr_upscale_to_x=A.hr_resize_x;A.hr_upscale_to_y=A.hr_resize_x*A.height//A.width
				else:A.hr_upscale_to_x=A.hr_resize_y*A.width//A.height;A.hr_upscale_to_y=A.hr_resize_y
				A.truncate_x=(A.hr_upscale_to_x-B)//opt_f;A.truncate_y=(A.hr_upscale_to_y-C)//opt_f
	def init(A,all_prompts,all_seeds,all_subseeds):
		if A.enable_hr:
			if A.hr_checkpoint_name:
				A.hr_checkpoint_info=sd_models.get_closet_checkpoint_match(A.hr_checkpoint_name)
				if A.hr_checkpoint_info is _A:raise Exception(f"Could not find checkpoint with name {A.hr_checkpoint_name}")
				A.extra_generation_params['Hires checkpoint']=A.hr_checkpoint_info.short_title
			if A.hr_sampler_name is not _A and A.hr_sampler_name!=A.sampler_name:A.extra_generation_params['Hires sampler']=A.hr_sampler_name
			if tuple(A.hr_prompt)!=tuple(A.prompt):A.extra_generation_params['Hires prompt']=A.hr_prompt
			if tuple(A.hr_negative_prompt)!=tuple(A.negative_prompt):A.extra_generation_params['Hires negative prompt']=A.hr_negative_prompt
			A.latent_scale_mode=shared.latent_upscale_modes.get(A.hr_upscaler,_A)if A.hr_upscaler is not _A else shared.latent_upscale_modes.get(shared.latent_upscale_default_mode,'nearest')
			if A.enable_hr and A.latent_scale_mode is _A:
				if not any(B.name==A.hr_upscaler for B in shared.sd_upscalers):raise Exception(f"could not find upscaler named {A.hr_upscaler}")
			A.calculate_target_resolution()
			if not state.processing_has_refined_job_count:
				if state.job_count==-1:state.job_count=A.n_iter
				shared.total_tqdm.updateTotal((A.steps+(A.hr_second_pass_steps or A.steps))*state.job_count);state.job_count=state.job_count*2;state.processing_has_refined_job_count=_C
			if A.hr_second_pass_steps:A.extra_generation_params['Hires steps']=A.hr_second_pass_steps
			if A.hr_upscaler is not _A:A.extra_generation_params['Hires upscaler']=A.hr_upscaler
	def sample(A,conditioning,unconditional_conditioning,seeds,subseeds,subseed_strength,prompts):
		A.sampler=sd_samplers.create_sampler(A.sampler_name,A.sd_model);B=A.rng.next();C=A.sampler.sample(A,B,conditioning,unconditional_conditioning,image_conditioning=A.txt2img_image_conditioning(B));del B
		if not A.enable_hr:return C
		if A.latent_scale_mode is _A:D=torch.stack(decode_latent_batch(A.sd_model,C,target_device=devices.cpu,check_for_nans=_C)).to(dtype=torch.float32)
		else:D=_A
		with sd_models.SkipWritingToConfig():sd_models.reload_model_weights(info=A.hr_checkpoint_info)
		devices.torch_gc();return A.sample_hr_pass(C,D,seeds,subseeds,subseed_strength,prompts)
	def sample_hr_pass(A,samples,decoded_samples,seeds,subseeds,subseed_strength,prompts):
		C=decoded_samples;B=samples
		if shared.state.interrupted:return B
		A.is_hr_pass=_C;H=A.hr_upscale_to_x;I=A.hr_upscale_to_y
		def J(image,index):
			'saves image before applying hires fix, if enabled in options; takes as an argument either an image or batch with latent space images';C=index;B=image
			if not A.save_samples()or not opts.save_images_before_highres_fix:return
			if not isinstance(B,Image.Image):B=sd_samplers.sample_to_image(B,C,approximation=0)
			D=create_infotext(A,A.all_prompts,A.all_seeds,A.all_subseeds,[],iteration=A.iteration,position_in_batch=C);images.save_image(B,A.outpath_samples,'',seeds[C],prompts[C],opts.samples_format,info=D,p=A,suffix='-before-highres-fix')
		L=A.hr_sampler_name or A.sampler_name;A.sampler=sd_samplers.create_sampler(L,A.sd_model)
		if A.latent_scale_mode is not _A:
			for F in range(B.shape[0]):J(B,F)
			B=torch.nn.functional.interpolate(B,size=(I//opt_f,H//opt_f),mode=A.latent_scale_mode['mode'],antialias=A.latent_scale_mode['antialias'])
			if getattr(A,_J,shared.opts.inpainting_mask_weight)<_D:G=A.img2img_image_conditioning(decode_first_stage(A.sd_model,B),B)
			else:G=A.txt2img_image_conditioning(B)
		else:
			M=torch.clamp((C+_D)/2.,min=.0,max=_D);K=[]
			for(F,E)in enumerate(M):E=255.*np.moveaxis(E.cpu().numpy(),0,2);E=E.astype(np.uint8);D=Image.fromarray(E);J(D,F);D=images.resize_image(0,D,H,I,upscaler_name=A.hr_upscaler);D=np.array(D).astype(np.float32)/255.;D=np.moveaxis(D,2,0);K.append(D)
			C=torch.from_numpy(np.array(K));C=C.to(shared.device,dtype=devices.dtype_vae)
			if opts.sd_vae_encode_method!=_L:A.extra_generation_params[_P]=opts.sd_vae_encode_method
			B=images_tensor_to_samples(C,approximation_indexes.get(opts.sd_vae_encode_method));G=A.img2img_image_conditioning(C,B)
		shared.state.nextjob();B=B[:,:,A.truncate_y//2:B.shape[2]-(A.truncate_y+1)//2,A.truncate_x//2:B.shape[3]-(A.truncate_x+1)//2];A.rng=rng.ImageRNG(B.shape[1:],A.seeds,subseeds=A.subseeds,subseed_strength=A.subseed_strength,seed_resize_from_h=A.seed_resize_from_h,seed_resize_from_w=A.seed_resize_from_w);N=A.rng.next();devices.torch_gc()
		if not A.disable_extra_networks:
			with devices.autocast():extra_networks.activate(A,A.hr_extra_network_data)
		with devices.autocast():A.calculate_hr_conds()
		sd_models.apply_token_merging(A.sd_model,A.get_token_merging_ratio(for_hr=_C))
		if A.scripts is not _A:A.scripts.before_hr(A)
		B=A.sampler.sample_img2img(A,B,N,A.hr_c,A.hr_uc,steps=A.hr_second_pass_steps or A.steps,image_conditioning=G);sd_models.apply_token_merging(A.sd_model,A.get_token_merging_ratio());A.sampler=_A;devices.torch_gc();C=decode_latent_batch(A.sd_model,B,target_device=devices.cpu,check_for_nans=_C);A.is_hr_pass=_B;return C
	def close(A):
		super().close();A.hr_c=_A;A.hr_uc=_A
		if not opts.persistent_cond_cache:StandardDemoProcessingTxt2Img.cached_hr_uc=[_A,_A];StandardDemoProcessingTxt2Img.cached_hr_c=[_A,_A]
	def setup_prompts(A):
		super().setup_prompts()
		if not A.enable_hr:return
		if A.hr_prompt=='':A.hr_prompt=A.prompt
		if A.hr_negative_prompt=='':A.hr_negative_prompt=A.negative_prompt
		if isinstance(A.hr_prompt,list):A.all_hr_prompts=A.hr_prompt
		else:A.all_hr_prompts=A.batch_size*A.n_iter*[A.hr_prompt]
		if isinstance(A.hr_negative_prompt,list):A.all_hr_negative_prompts=A.hr_negative_prompt
		else:A.all_hr_negative_prompts=A.batch_size*A.n_iter*[A.hr_negative_prompt]
		A.all_hr_prompts=[shared.prompt_styles.apply_styles_to_prompt(B,A.styles)for B in A.all_hr_prompts];A.all_hr_negative_prompts=[shared.prompt_styles.apply_negative_styles_to_prompt(B,A.styles)for B in A.all_hr_negative_prompts]
	def calculate_hr_conds(A):
		if A.hr_c is not _A:return
		E=prompt_parser.SdConditioning(A.hr_prompts,width=A.hr_upscale_to_x,height=A.hr_upscale_to_y);F=prompt_parser.SdConditioning(A.hr_negative_prompts,width=A.hr_upscale_to_x,height=A.hr_upscale_to_y,is_negative_prompt=_C);B=sd_samplers.find_sampler_config(A.hr_sampler_name or A.sampler_name);C=A.hr_second_pass_steps or A.steps;D=B.total_steps(C)if B else C;A.hr_uc=A.get_conds_with_caching(prompt_parser.get_learned_conditioning,F,A.firstpass_steps,[A.cached_hr_uc,A.cached_uc],A.hr_extra_network_data,D);A.hr_c=A.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning,E,A.firstpass_steps,[A.cached_hr_c,A.cached_c],A.hr_extra_network_data,D)
	def setup_conds(A):
		if A.is_hr_pass:A.hr_c=_A;A.calculate_hr_conds();return
		super().setup_conds();A.hr_uc=_A;A.hr_c=_A
		if A.enable_hr and A.hr_checkpoint_info is _A:
			if shared.opts.hires_fix_use_firstpass_conds:A.calculate_hr_conds()
			elif lowvram.is_enabled(shared.sd_model)and shared.sd_model.sd_checkpoint_info==sd_models.select_checkpoint():
				with devices.autocast():extra_networks.activate(A,A.hr_extra_network_data)
				A.calculate_hr_conds()
				with devices.autocast():extra_networks.activate(A,A.extra_network_data)
	def get_conds(A):
		if A.is_hr_pass:return A.hr_c,A.hr_uc
		return super().get_conds()
	def parse_extra_network_prompts(A):
		B=super().parse_extra_network_prompts()
		if A.enable_hr:A.hr_prompts=A.all_hr_prompts[A.iteration*A.batch_size:(A.iteration+1)*A.batch_size];A.hr_negative_prompts=A.all_hr_negative_prompts[A.iteration*A.batch_size:(A.iteration+1)*A.batch_size];A.hr_prompts,A.hr_extra_network_data=extra_networks.parse_prompts(A.hr_prompts)
		return B
@dataclass(repr=_B)
class StandardDemoProcessingImg2Img(StandardDemoProcessing):
	init_images:list=_A;resize_mode:int=0;denoising_strength:float=.75;image_cfg_scale:float=_A;mask:Any=_A;mask_blur_x:int=4;mask_blur_y:int=4;mask_blur:int=_A;inpainting_fill:int=0;inpaint_full_res:bool=_C;inpaint_full_res_padding:int=0;inpainting_mask_invert:int=0;initial_noise_multiplier:float=_A;latent_mask:Image=_A;image_mask:Any=field(default=_A,init=_B);nmask:torch.Tensor=field(default=_A,init=_B);image_conditioning:torch.Tensor=field(default=_A,init=_B);init_img_hash:str=field(default=_A,init=_B);mask_for_overlay:Image=field(default=_A,init=_B);init_latent:torch.Tensor=field(default=_A,init=_B)
	def __post_init__(A):super().__post_init__();A.image_mask=A.mask;A.mask=_A;A.initial_noise_multiplier=opts.initial_noise_multiplier if A.initial_noise_multiplier is _A else A.initial_noise_multiplier
	@property
	def mask_blur(self):
		A=self
		if A.mask_blur_x==A.mask_blur_y:return A.mask_blur_x
	@mask_blur.setter
	def mask_blur(self,value):
		A=value
		if isinstance(A,int):self.mask_blur_x=A;self.mask_blur_y=A
	def init(A,all_prompts,all_seeds,all_subseeds):
		A.image_cfg_scale=A.image_cfg_scale if shared.sd_model.cond_stage_key=='edit'else _A;A.sampler=sd_samplers.create_sampler(A.sampler_name,A.sd_model);F=_A;C=A.image_mask
		if C is not _A:
			C=create_binary_mask(C)
			if A.inpainting_mask_invert:C=ImageOps.invert(C)
			if A.mask_blur_x>0:D=np.array(C);I=2*int(2.5*A.mask_blur_x+.5)+1;D=cv2.GaussianBlur(D,(I,1),A.mask_blur_x);C=Image.fromarray(D)
			if A.mask_blur_y>0:D=np.array(C);I=2*int(2.5*A.mask_blur_y+.5)+1;D=cv2.GaussianBlur(D,(1,I),A.mask_blur_y);C=Image.fromarray(D)
			if A.inpaint_full_res:A.mask_for_overlay=C;H=C.convert('L');F=masking.get_crop_region(np.array(H),A.inpaint_full_res_padding);F=masking.expand_crop_region(F,A.width,A.height,H.width,H.height);K,L,Q,R=F;H=H.crop(F);C=images.resize_image(2,H,A.width,A.height);A.paste_to=K,L,Q-K,R-L
			else:C=images.resize_image(A.resize_mode,C,A.width,A.height);D=np.array(C);D=np.clip(D.astype(np.float32)*2,0,255).astype(np.uint8);A.mask_for_overlay=Image.fromarray(D)
			A.overlay_images=[]
		M=A.latent_mask if A.latent_mask is not _A else C;N=opts.img2img_color_correction and A.color_corrections is _A
		if N:A.color_corrections=[]
		G=[]
		for J in A.init_images:
			if opts.save_init_img:A.init_img_hash=hashlib.md5(J.tobytes()).hexdigest();images.save_image(J,path=opts.outdir_init_images,basename=_A,forced_filename=A.init_img_hash,save_to_dirs=_B)
			B=images.flatten(J,opts.img2img_background_color)
			if F is _A and A.resize_mode!=3:B=images.resize_image(A.resize_mode,B,A.width,A.height)
			if C is not _A:O=Image.new(_G,(B.width,B.height));O.paste(B.convert(_E).convert(_G),mask=ImageOps.invert(A.mask_for_overlay.convert('L')));A.overlay_images.append(O.convert(_E))
			if F is not _A:B=B.crop(F);B=images.resize_image(2,B,A.width,A.height)
			if C is not _A:
				if A.inpainting_fill!=1:B=masking.fill(B,M)
			if N:A.color_corrections.append(setup_color_correction(B))
			B=np.array(B).astype(np.float32)/255.;B=np.moveaxis(B,2,0);G.append(B)
		if len(G)==1:
			P=np.expand_dims(G[0],axis=0).repeat(A.batch_size,axis=0)
			if A.overlay_images is not _A:A.overlay_images=A.overlay_images*A.batch_size
			if A.color_corrections is not _A and len(A.color_corrections)==1:A.color_corrections=A.color_corrections*A.batch_size
		elif len(G)<=A.batch_size:A.batch_size=len(G);P=np.array(G)
		else:raise RuntimeError(f"bad number of images passed: {len(G)}; expecting {A.batch_size} or less")
		B=torch.from_numpy(P);B=B.to(shared.device,dtype=devices.dtype_vae)
		if opts.sd_vae_encode_method!=_L:A.extra_generation_params[_P]=opts.sd_vae_encode_method
		A.init_latent=images_tensor_to_samples(B,approximation_indexes.get(opts.sd_vae_encode_method),A.sd_model);devices.torch_gc()
		if A.resize_mode==3:A.init_latent=torch.nn.functional.interpolate(A.init_latent,size=(A.height//opt_f,A.width//opt_f),mode='bilinear')
		if C is not _A:
			S=M;E=S.convert(_F).resize((A.init_latent.shape[3],A.init_latent.shape[2]));E=np.moveaxis(np.array(E,dtype=np.float32),2,0)/255;E=E[0];E=np.around(E);E=np.tile(E[_A],(4,1,1));A.mask=torch.asarray(_D-E).to(shared.device).type(A.sd_model.dtype);A.nmask=torch.asarray(E).to(shared.device).type(A.sd_model.dtype)
			if A.inpainting_fill==2:A.init_latent=A.init_latent*A.mask+create_random_tensors(A.init_latent.shape[1:],all_seeds[0:A.init_latent.shape[0]])*A.nmask
			elif A.inpainting_fill==3:A.init_latent=A.init_latent*A.mask
		A.image_conditioning=A.img2img_image_conditioning(B*2-1,A.init_latent,C)
	def sample(A,conditioning,unconditional_conditioning,seeds,subseeds,subseed_strength,prompts):
		B=A.rng.next()
		if A.initial_noise_multiplier!=_D:A.extra_generation_params['Noise multiplier']=A.initial_noise_multiplier;B*=A.initial_noise_multiplier
		C=A.sampler.sample_img2img(A,A.init_latent,B,conditioning,unconditional_conditioning,image_conditioning=A.image_conditioning)
		if A.mask is not _A:C=C*A.nmask+A.init_latent*A.mask
		del B;devices.torch_gc();return C
	def get_token_merging_ratio(A,for_hr=_B):return A.token_merging_ratio or'token_merging_ratio'in A.override_settings and opts.token_merging_ratio or opts.token_merging_ratio_img2img or opts.token_merging_ratio