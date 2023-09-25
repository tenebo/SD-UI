_B='configs'
_A=None
import os,torch
from modules import shared,paths,sd_disable_initialization,devices
sd_configs_path=shared.sd_configs_path
sd_repo_configs_path=os.path.join(paths.paths['Stable Diffusion'],_B,'stable-diffusion')
sd_xl_repo_configs_path=os.path.join(paths.paths['Stable Diffusion XL'],_B,'inference')
config_default=shared.sd_default_config
config_sd2=os.path.join(sd_repo_configs_path,'v2-inference.yaml')
config_sd2v=os.path.join(sd_repo_configs_path,'v2-inference-v.yaml')
config_sd2_inpainting=os.path.join(sd_repo_configs_path,'v2-inpainting-inference.yaml')
config_sdxl=os.path.join(sd_xl_repo_configs_path,'sd_xl_base.yaml')
config_sdxl_refiner=os.path.join(sd_xl_repo_configs_path,'sd_xl_refiner.yaml')
config_depth_model=os.path.join(sd_repo_configs_path,'v2-midas-inference.yaml')
config_unclip=os.path.join(sd_repo_configs_path,'v2-1-stable-unclip-l-inference.yaml')
config_unopenclip=os.path.join(sd_repo_configs_path,'v2-1-stable-unclip-h-inference.yaml')
config_inpainting=os.path.join(sd_configs_path,'v1-inpainting-inference.yaml')
config_instruct_pix2pix=os.path.join(sd_configs_path,'instruct-pix2pix.yaml')
config_alt_diffusion=os.path.join(sd_configs_path,'alt-diffusion-inference.yaml')
def is_using_v_parameterization_for_sd2(state_dict):
	"\n    Detects whether unet in state_dict is using v-parameterization. Returns True if it is. You're welcome.\n    ";F='model.diffusion_model.';E=False;C=True;import ldm.modules.diffusionmodules.openaimodel;A=devices.cpu
	with sd_disable_initialization.DisableInitialization():B=ldm.modules.diffusionmodules.openaimodel.UNetModel(use_checkpoint=C,use_fp16=E,image_size=32,in_channels=4,out_channels=4,model_channels=320,attention_resolutions=[4,2,1],num_res_blocks=2,channel_mult=[1,2,4,4],num_head_channels=64,use_spatial_transformer=C,use_linear_in_transformer=C,transformer_depth=1,context_dim=1024,legacy=E);B.eval()
	with torch.no_grad():G={A.replace(F,''):B for(A,B)in state_dict.items()if F in A};B.load_state_dict(G,strict=C);B.to(device=A,dtype=torch.float);H=torch.ones((1,2,1024),device=A)*.5;D=torch.ones((1,4,8,8),device=A)*.5;I=(B(D,torch.asarray([999],device=A),context=H)-D).mean().item()
	return I<-1
def guess_model_config_from_state_dict(sd,filename):
	A=sd;D=A.get('cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight',_A);B=A.get('model.diffusion_model.input_blocks.0.0.weight',_A);C=A.get('embedder.model.ln_final.weight',_A)
	if A.get('conditioner.embedders.1.model.ln_final.weight',_A)is not _A:return config_sdxl
	if A.get('conditioner.embedders.0.model.ln_final.weight',_A)is not _A:return config_sdxl_refiner
	elif A.get('depth_model.model.pretrained.act_postprocess3.0.project.0.bias',_A)is not _A:return config_depth_model
	elif C is not _A and C.shape[0]==768:return config_unclip
	elif C is not _A and C.shape[0]==1024:return config_unopenclip
	if D is not _A and D.shape[1]==1024:
		if B.shape[1]==9:return config_sd2_inpainting
		elif is_using_v_parameterization_for_sd2(A):return config_sd2v
		else:return config_sd2
	if B is not _A:
		if B.shape[1]==9:return config_inpainting
		if B.shape[1]==8:return config_instruct_pix2pix
	if A.get('cond_stage_model.roberta.embeddings.word_embeddings.weight',_A)is not _A:return config_alt_diffusion
	return config_default
def find_checkpoint_config(state_dict,info):
	B=state_dict;A=info
	if A is _A:return guess_model_config_from_state_dict(B,'')
	C=find_checkpoint_config_near_filename(A)
	if C is not _A:return C
	return guess_model_config_from_state_dict(B,A.filename)
def find_checkpoint_config_near_filename(info):
	if info is _A:return
	A=f"{os.path.splitext(info.filename)[0]}.yaml"
	if os.path.exists(A):return A