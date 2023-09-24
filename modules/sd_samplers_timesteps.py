_I='Pad conds'
_H='s_min_uncond'
_G='cond_scale'
_F='uncond'
_E='image_cond'
_D=False
_C='timesteps'
_B=True
_A=None
import torch,inspect,sys
from modules import devices,sd_samplers_common,sd_samplers_timesteps_impl
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
from modules.script_callbacks import ExtraNoiseParams,extra_noise_callback
from modules.shared import opts
import modules.shared as shared
samplers_timesteps=[('DDIM',sd_samplers_timesteps_impl.ddim,['ddim'],{}),('PLMS',sd_samplers_timesteps_impl.plms,['plms'],{}),('UniPC',sd_samplers_timesteps_impl.unipc,['unipc'],{})]
samplers_data_timesteps=[sd_samplers_common.SamplerData(A,lambda model,funcname=B:CompVisSampler(funcname,model),C,D)for(A,B,C,D)in samplers_timesteps]
class CompVisTimestepsDenoiser(torch.nn.Module):
	def __init__(A,model,*B,**C):super().__init__(*B,**C);A.inner_model=model
	def forward(A,input,timesteps,**B):return A.inner_model.apply_model(input,timesteps,**B)
class CompVisTimestepsVDenoiser(torch.nn.Module):
	def __init__(A,model,*B,**C):super().__init__(*B,**C);A.inner_model=model
	def predict_eps_from_z_and_v(A,x_t,t,v):return A.inner_model.sqrt_alphas_cumprod[t.to(torch.int),_A,_A,_A]*v+A.inner_model.sqrt_one_minus_alphas_cumprod[t.to(torch.int),_A,_A,_A]*x_t
	def forward(A,input,timesteps,**C):B=timesteps;D=A.inner_model.apply_model(input,B,**C);E=A.predict_eps_from_z_and_v(input,B,D);return E
class CFGDenoiserTimesteps(CFGDenoiser):
	def __init__(A,sampler):super().__init__(sampler);A.alphas=shared.sd_model.alphas_cumprod;A.mask_before_denoising=_B
	def get_pred_x0(B,x_in,x_out,sigma):C=sigma.to(dtype=int);A=B.alphas[C][:,_A,_A,_A];D=(1-A).sqrt();E=(x_in-D*x_out)/A.sqrt();return E
	@property
	def inner_model(self):
		A=self
		if A.model_wrap is _A:B=CompVisTimestepsVDenoiser if shared.sd_model.parameterization=='v'else CompVisTimestepsDenoiser;A.model_wrap=B(shared.sd_model)
		return A.model_wrap
class CompVisSampler(sd_samplers_common.Sampler):
	def __init__(A,funcname,sd_model):super().__init__(funcname);A.eta_option_field='eta_ddim';A.eta_infotext_field='Eta DDIM';A.eta_default=.0;A.model_wrap_cfg=CFGDenoiserTimesteps(A)
	def get_timesteps(B,p,steps):
		C=steps;A=B.config is not _A and B.config.options.get('discard_next_to_last_sigma',_D)
		if opts.always_discard_next_to_last_sigma and not A:A=_B;p.extra_generation_params['Discard penultimate sigma']=_B
		C+=1 if A else 0;D=torch.clip(torch.asarray(list(range(0,1000,1000//C)),device=devices.device)+1,0,999);return D
	def sample_img2img(A,p,x,noise,conditioning,unconditional_conditioning,steps=_A,image_conditioning=_A):
		L='is_img2img';D=steps;B=noise;D,C=sd_samplers_common.setup_img2img_steps(p,D);E=A.get_timesteps(p,D);M=E[:C];H=shared.sd_model.alphas_cumprod;I=torch.sqrt(H[E[C]]);N=torch.sqrt(1-H[E[C]]);F=x*I+B*N
		if opts.img2img_extra_noise>0:p.extra_generation_params['Extra noise']=opts.img2img_extra_noise;J=ExtraNoiseParams(B,x,F);extra_noise_callback(J);B=J.noise;F+=B*opts.img2img_extra_noise*I
		G=A.initialize(p);K=inspect.signature(A.func).parameters
		if _C in K:G[_C]=M
		if L in K:G[L]=_B
		A.model_wrap_cfg.init_latent=x;A.last_latent=x;A.sampler_extra_args={'cond':conditioning,_E:image_conditioning,_F:unconditional_conditioning,_G:p.cfg_scale,_H:A.s_min_uncond};O=A.launch_sampling(C+1,lambda:A.func(A.model_wrap_cfg,F,extra_args=A.sampler_extra_args,disable=_D,callback=A.callback_state,**G))
		if A.model_wrap_cfg.padded_cond_uncond:p.extra_generation_params[_I]=_B
		return O
	def sample(A,p,x,conditioning,unconditional_conditioning,steps=_A,image_conditioning=_A):
		B=steps;B=B or p.steps;D=A.get_timesteps(p,B);C=A.initialize(p);E=inspect.signature(A.func).parameters
		if _C in E:C[_C]=D
		A.last_latent=x;A.sampler_extra_args={'cond':conditioning,_E:image_conditioning,_F:unconditional_conditioning,_G:p.cfg_scale,_H:A.s_min_uncond};F=A.launch_sampling(B,lambda:A.func(A.model_wrap_cfg,x,extra_args=A.sampler_extra_args,disable=_D,callback=A.callback_state,**C))
		if A.model_wrap_cfg.padded_cond_uncond:p.extra_generation_params[_I]=_B
		return F
sys.modules['modules.sd_samplers_compvis']=sys.modules[__name__]
VanillaStandardDemoSampler=CompVisSampler