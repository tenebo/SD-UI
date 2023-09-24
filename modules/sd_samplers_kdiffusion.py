_m='Pad conds'
_l='s_min_uncond'
_k='cond_scale'
_j='uncond'
_i='image_cond'
_h='noise_sampler'
_g='polyexponential'
_f='Automatic'
_e='sample_dpm_fast'
_d='sample_heun'
_c='sample_lms'
_b='sample_euler'
_a='k_dpmpp_2m_sde_ka'
_Z='sample_dpmpp_2m'
_Y='s_tmax'
_X='s_tmin'
_W='s_churn'
_V='sample_dpmpp_2s_ancestral'
_U='sample_dpm_2_ancestral'
_T='sample_dpm_2'
_S='sample_dpmpp_sde'
_R='sigmas'
_Q='sample_dpmpp_3m_sde'
_P='sigma_max'
_O=False
_N='sigma_min'
_M='exponential'
_L='heun'
_K='solver_type'
_J='sample_dpmpp_2m_sde'
_I='discard_next_to_last_sigma'
_H='uses_ensd'
_G='s_noise'
_F='second_order'
_E='karras'
_D='brownian_noise'
_C=None
_B='scheduler'
_A=True
import torch,inspect,k_diffusion.sampling
from modules import sd_samplers_common,sd_samplers_extra,sd_samplers_cfg_denoiser
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
from modules.script_callbacks import ExtraNoiseParams,extra_noise_callback
from modules.shared import opts
import modules.shared as shared
samplers_k_diffusion=[('DPM++ 2M Karras',_Z,['k_dpmpp_2m_ka'],{_B:_E}),('DPM++ SDE Karras',_S,['k_dpmpp_sde_ka'],{_B:_E,_F:_A,_D:_A}),('DPM++ 2M SDE Exponential',_J,['k_dpmpp_2m_sde_exp'],{_B:_M,_D:_A}),('DPM++ 2M SDE Karras',_J,[_a],{_B:_E,_D:_A}),('Euler a','sample_euler_ancestral',['k_euler_a','k_euler_ancestral'],{_H:_A}),('Euler',_b,['k_euler'],{}),('LMS',_c,['k_lms'],{}),('Heun',_d,['k_heun'],{_F:_A}),('DPM2',_T,['k_dpm_2'],{_I:_A,_F:_A}),('DPM2 a',_U,['k_dpm_2_a'],{_I:_A,_H:_A,_F:_A}),('DPM++ 2S a',_V,['k_dpmpp_2s_a'],{_H:_A,_F:_A}),('DPM++ 2M',_Z,['k_dpmpp_2m'],{}),('DPM++ SDE',_S,['k_dpmpp_sde'],{_F:_A,_D:_A}),('DPM++ 2M SDE',_J,[_a],{_D:_A}),('DPM++ 2M SDE Heun',_J,['k_dpmpp_2m_sde_heun'],{_D:_A,_K:_L}),('DPM++ 2M SDE Heun Karras',_J,['k_dpmpp_2m_sde_heun_ka'],{_B:_E,_D:_A,_K:_L}),('DPM++ 2M SDE Heun Exponential',_J,['k_dpmpp_2m_sde_heun_exp'],{_B:_M,_D:_A,_K:_L}),('DPM++ 3M SDE',_Q,['k_dpmpp_3m_sde'],{_I:_A,_D:_A}),('DPM++ 3M SDE Karras',_Q,['k_dpmpp_3m_sde_ka'],{_B:_E,_I:_A,_D:_A}),('DPM++ 3M SDE Exponential',_Q,['k_dpmpp_3m_sde_exp'],{_B:_M,_I:_A,_D:_A}),('DPM fast',_e,['k_dpm_fast'],{_H:_A}),('DPM adaptive','sample_dpm_adaptive',['k_dpm_ad'],{_H:_A}),('LMS Karras',_c,['k_lms_ka'],{_B:_E}),('DPM2 Karras',_T,['k_dpm_2_ka'],{_B:_E,_I:_A,_H:_A,_F:_A}),('DPM2 a Karras',_U,['k_dpm_2_a_ka'],{_B:_E,_I:_A,_H:_A,_F:_A}),('DPM++ 2S a Karras',_V,['k_dpmpp_2s_a_ka'],{_B:_E,_H:_A,_F:_A}),('Restart',sd_samplers_extra.restart_sampler,['restart'],{_B:_E,_F:_A})]
samplers_data_k_diffusion=[sd_samplers_common.SamplerData(B,lambda model,funcname=A:KDiffusionSampler(funcname,model),C,D)for(B,A,C,D)in samplers_k_diffusion if callable(A)or hasattr(k_diffusion.sampling,A)]
sampler_extra_params={_b:[_W,_X,_Y,_G],_d:[_W,_X,_Y,_G],_T:[_W,_X,_Y,_G],_e:[_G],_U:[_G],_V:[_G],_S:[_G],_J:[_G],_Q:[_G]}
k_diffusion_samplers_map={A.name:A for A in samplers_data_k_diffusion}
k_diffusion_scheduler={_f:_C,_E:k_diffusion.sampling.get_sigmas_karras,_M:k_diffusion.sampling.get_sigmas_exponential,_g:k_diffusion.sampling.get_sigmas_polyexponential}
class CFGDenoiserKDiffusion(sd_samplers_cfg_denoiser.CFGDenoiser):
	@property
	def inner_model(self):
		A=self
		if A.model_wrap is _C:B=k_diffusion.external.CompVisVDenoiser if shared.sd_model.parameterization=='v'else k_diffusion.external.CompVisDenoiser;A.model_wrap=B(shared.sd_model,quantize=shared.opts.enable_quantization)
		return A.model_wrap
class KDiffusionSampler(sd_samplers_common.Sampler):
	def __init__(A,funcname,sd_model,options=_C):B=funcname;super().__init__(B);A.extra_params=sampler_extra_params.get(B,[]);A.options=options or{};A.func=B if callable(B)else getattr(k_diffusion.sampling,A.funcname);A.model_wrap_cfg=CFGDenoiserKDiffusion(A);A.model_wrap=A.model_wrap_cfg.inner_model
	def get_sigmas(A,p,steps):
		C=steps;D=A.config is not _C and A.config.options.get(_I,_O)
		if opts.always_discard_next_to_last_sigma and not D:D=_A;p.extra_generation_params['Discard penultimate sigma']=_A
		C+=1 if D else 0
		if p.sampler_noise_scheduler_override:B=p.sampler_noise_scheduler_override(C)
		elif opts.k_sched_type!=_f:
			E,F=A.model_wrap.sigmas[0].item(),A.model_wrap.sigmas[-1].item();H,I=(.1,10)if opts.use_old_karras_scheduler_sigmas else(E,F);G={_N:H,_P:I};J=k_diffusion_scheduler[opts.k_sched_type];p.extra_generation_params['Schedule type']=opts.k_sched_type
			if opts.sigma_min!=E and opts.sigma_min!=0:G[_N]=opts.sigma_min;p.extra_generation_params['Schedule min sigma']=opts.sigma_min
			if opts.sigma_max!=F and opts.sigma_max!=0:G[_P]=opts.sigma_max;p.extra_generation_params['Schedule max sigma']=opts.sigma_max
			K=1. if opts.k_sched_type==_g else 7.
			if opts.k_sched_type!=_M and opts.rho!=0 and opts.rho!=K:G['rho']=opts.rho;p.extra_generation_params['Schedule rho']=opts.rho
			B=J(n=C,**G,device=shared.device)
		elif A.config is not _C and A.config.options.get(_B,_C)==_E:H,I=(.1,10)if opts.use_old_karras_scheduler_sigmas else(A.model_wrap.sigmas[0].item(),A.model_wrap.sigmas[-1].item());B=k_diffusion.sampling.get_sigmas_karras(n=C,sigma_min=H,sigma_max=I,device=shared.device)
		elif A.config is not _C and A.config.options.get(_B,_C)==_M:E,F=A.model_wrap.sigmas[0].item(),A.model_wrap.sigmas[-1].item();B=k_diffusion.sampling.get_sigmas_exponential(n=C,sigma_min=E,sigma_max=F,device=shared.device)
		else:B=A.model_wrap.get_sigmas(C)
		if D:B=torch.cat([B[:-2],B[-1:]])
		return B
	def sample_img2img(A,p,x,noise,conditioning,unconditional_conditioning,steps=_C,image_conditioning=_C):
		H='sigma_sched';E=steps;F=noise;E,I=sd_samplers_common.setup_img2img_steps(p,E);J=A.get_sigmas(p,E);C=J[E-I-1:];G=x+F*C[0]
		if opts.img2img_extra_noise>0:p.extra_generation_params['Extra noise']=opts.img2img_extra_noise;K=ExtraNoiseParams(F,x,G);extra_noise_callback(K);F=K.noise;G+=F*opts.img2img_extra_noise
		B=A.initialize(p);D=inspect.signature(A.func).parameters
		if _N in D:B[_N]=C[-2]
		if _P in D:B[_P]=C[0]
		if'n'in D:B['n']=len(C)-1
		if H in D:B[H]=C
		if _R in D:B[_R]=C
		if A.config.options.get(_D,_O):L=A.create_noise_sampler(x,J,p);B[_h]=L
		if A.config.options.get(_K,_C)==_L:B[_K]=_L
		A.model_wrap_cfg.init_latent=x;A.last_latent=x;A.sampler_extra_args={'cond':conditioning,_i:image_conditioning,_j:unconditional_conditioning,_k:p.cfg_scale,_l:A.s_min_uncond};M=A.launch_sampling(I+1,lambda:A.func(A.model_wrap_cfg,G,extra_args=A.sampler_extra_args,disable=_O,callback=A.callback_state,**B))
		if A.model_wrap_cfg.padded_cond_uncond:p.extra_generation_params[_m]=_A
		return M
	def sample(A,p,x,conditioning,unconditional_conditioning,steps=_C,image_conditioning=_C):
		C=steps;C=C or p.steps;D=A.get_sigmas(p,C)
		if opts.sgm_noise_multiplier:p.extra_generation_params['SGM noise multiplier']=_A;x=x*torch.sqrt(1.+D[0]**2.)
		else:x=x*D[0]
		B=A.initialize(p);E=inspect.signature(A.func).parameters
		if'n'in E:B['n']=C
		if _N in E:B[_N]=A.model_wrap.sigmas[0].item();B[_P]=A.model_wrap.sigmas[-1].item()
		if _R in E:B[_R]=D
		if A.config.options.get(_D,_O):F=A.create_noise_sampler(x,D,p);B[_h]=F
		if A.config.options.get(_K,_C)==_L:B[_K]=_L
		A.last_latent=x;A.sampler_extra_args={'cond':conditioning,_i:image_conditioning,_j:unconditional_conditioning,_k:p.cfg_scale,_l:A.s_min_uncond};G=A.launch_sampling(C,lambda:A.func(A.model_wrap_cfg,x,extra_args=A.sampler_extra_args,disable=_O,callback=A.callback_state,**B))
		if A.model_wrap_cfg.padded_cond_uncond:p.extra_generation_params[_m]=_A
		return G