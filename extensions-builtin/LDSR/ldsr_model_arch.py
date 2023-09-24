_F='sample'
_E='split_input_params'
_D=True
_C=False
_B=1.
_A=None
import os,gc,time,numpy as np,torch,torchvision
from PIL import Image
from einops import rearrange,repeat
from omegaconf import OmegaConf
import safetensors.torch
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config,ismap
from modules import shared,sd_hijack,devices
cached_ldsr_model=_A
class LDSR:
	def load_model_from_config(B,half_attention):
		D='state_dict';E='cpu';global cached_ldsr_model
		if shared.opts.ldsr_cached and cached_ldsr_model is not _A:print('Loading model from cache');A=cached_ldsr_model
		else:
			print(f"Loading model from {B.modelPath}");I,G=os.path.splitext(B.modelPath)
			if G.lower()=='.safetensors':C=safetensors.torch.load_file(B.modelPath,device=E)
			else:C=torch.load(B.modelPath,map_location=E)
			H=C[D]if D in C else C;F=OmegaConf.load(B.yamlPath);F.model.target='ldm.models.diffusion.ddpm.LatentDiffusionV1';A=instantiate_from_config(F.model);A.load_state_dict(H,strict=_C);A=A.to(shared.device)
			if half_attention:A=A.half()
			if shared.cmd_opts.opt_channelslast:A=A.to(memory_format=torch.channels_last)
			sd_hijack.model_hijack.hijack(A);A.eval()
			if shared.opts.ldsr_cached:cached_ldsr_model=A
		return{'model':A}
	def __init__(A,model_path,yaml_path):A.modelPath=model_path;A.yamlPath=yaml_path
	@staticmethod
	def run(model,selected_path,custom_steps,eta):
		D=eta;B=model;E=get_cond(selected_path);I=1;J=_A;K=_A;L=_C;M=_B;D=D;A=_A;N,O=E['image'].shape[1:3];P=N>=128 and O>=128
		if P:F=128;G=64;Q=4;B.split_input_params={'ks':(F,F),'stride':(G,G),'vqf':Q,'patch_distributed_vq':_D,'tie_braker':_C,'clip_max_weight':.5,'clip_min_weight':.01,'clip_max_tie_weight':.5,'clip_min_tie_weight':.01}
		elif hasattr(B,_E):delattr(B,_E)
		C=_A;H=_A
		for R in range(I):
			if A is not _A:C=torch.randn(1,A[1],A[2],A[3]).to(B.device);C=repeat(C,'1 c h w -> b c h w',b=A[0])
			H=make_convolutional_sample(E,B,custom_steps=custom_steps,eta=D,quantize_x0=_C,custom_shape=A,temperature=M,noise_dropout=.0,corrector=J,corrector_kwargs=K,x_T=C,ddim_use_x0_pred=L)
		return H
	def super_resolution(E,image,steps=100,target_scale=2,half_attention=_C):
		F=target_scale;G=E.load_model_from_config(half_attention);L=int(steps);M=_B;gc.collect();devices.torch_gc();B=image;H,I=B.size;C=F/4;N=H*C;O=I*C;J=int(np.ceil(N));K=int(np.ceil(O))
		if C!=1:print(f"Downsampling from [{H}, {I}] to [{J}, {K}]");B=B.resize((J,K),Image.LANCZOS)
		else:print(f"Down sample rate is 1 from {F} / 4 (Not downsampling)")
		P,Q=np.max(((2,2),np.ceil(np.array(B.size)/64).astype(int)),axis=0)*64-B.size;R=Image.fromarray(np.pad(np.array(B),((0,Q),(0,P),(0,0)),mode='edge'));S=E.run(G['model'],R,L,M);A=S[_F];A=A.detach().cpu();A=torch.clamp(A,-_B,_B);A=(A+_B)/2.*255;A=A.numpy().astype(np.uint8);A=np.transpose(A,(0,2,3,1));D=Image.fromarray(A[0]);D=D.crop((0,0)+tuple(np.array(B.size)*4));del G;gc.collect();devices.torch_gc();return D
def get_cond(selected_path):D='1 c h w -> 1 h w c';B={};E=4;A=selected_path.convert('RGB');A=torch.unsqueeze(torchvision.transforms.ToTensor()(A),0);C=torchvision.transforms.functional.resize(A,size=[E*A.shape[2],E*A.shape[3]],antialias=_D);C=rearrange(C,D);A=rearrange(A,D);A=2.*A-_B;A=A.to(shared.device);B['LR_image']=A;B['image']=C;return B
@torch.no_grad()
def convsample_ddim(model,cond,steps,shape,eta=_B,callback=_A,normals_sequence=_A,mask=_A,x0=_A,quantize_x0=_C,temperature=_B,score_corrector=_A,corrector_kwargs=_A,x_t=_A):B=steps;A=shape;C=DDIMSampler(model);D=A[0];A=A[1:];print(f"Sampling with eta = {eta}; steps: {B}");E,F=C.sample(B,batch_size=D,shape=A,conditioning=cond,callback=callback,normals_sequence=normals_sequence,quantize_x0=quantize_x0,eta=eta,mask=mask,x0=x0,temperature=temperature,verbose=_C,score_corrector=score_corrector,corrector_kwargs=corrector_kwargs,x_t=x_t);return E,F
@torch.no_grad()
def make_convolutional_sample(batch,model,custom_steps=_A,eta=_B,quantize_x0=_C,custom_shape=_A,temperature=_B,noise_dropout=.0,corrector=_A,corrector_kwargs=_A,x_T=_A,ddim_use_x0_pred=_C):
	G='original_conditioning';D=custom_shape;A=model;B={};H,K,E,L,C=A.get_input(batch,A.first_stage_key,return_first_stage_outputs=_D,force_c_encode=not(hasattr(A,_E)and A.cond_stage_key=='coordinates_bbox'),return_original_cond=_D)
	if D is not _A:H=torch.randn(D);print(f"Generating {D[0]} samples of shape {D[1:]}")
	M=_A;B['input']=E;B['reconstruction']=L
	if ismap(C):
		B[G]=A.to_rgb(C)
		if hasattr(A,'cond_stage_key'):B[A.cond_stage_key]=A.to_rgb(C)
	else:
		B[G]=C if C is not _A else torch.zeros_like(E)
		if A.cond_stage_model:
			B[A.cond_stage_key]=C if C is not _A else torch.zeros_like(E)
			if A.cond_stage_key=='class_label':B[A.cond_stage_key]=C[A.cond_stage_key]
	with A.ema_scope('Plotting'):
		N=time.time();F,O=convsample_ddim(A,K,steps=custom_steps,shape=H.shape,eta=eta,quantize_x0=quantize_x0,mask=_A,x0=M,temperature=temperature,score_corrector=corrector,corrector_kwargs=corrector_kwargs,x_t=x_T);P=time.time()
		if ddim_use_x0_pred:F=O['pred_x0'][-1]
	I=A.decode_first_stage(F)
	try:J=A.decode_first_stage(F,force_not_quantize=_D);B['sample_noquant']=J;B['sample_diff']=torch.abs(J-I)
	except Exception:pass
	B[_F]=I;B['time']=P-N;return B