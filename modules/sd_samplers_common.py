_B=False
_A=None
import inspect
from collections import namedtuple
import numpy as np,torch
from PIL import Image
from modules import devices,images,sd_vae_approx,sd_samplers,sd_vae_taesd,shared,sd_models
from modules.shared import opts,state
import k_diffusion.sampling
SamplerDataTuple=namedtuple('SamplerData',['name','constructor','aliases','options'])
class SamplerData(SamplerDataTuple):
	def total_steps(B,steps):
		A=steps
		if B.options.get('second_order',_B):A=A*2
		return A
def setup_img2img_steps(p,steps=_A):
	A=steps
	if opts.img2img_fix_steps or A is not _A:B=A or p.steps;A=int(B/min(p.denoising_strength,.999))if p.denoising_strength>0 else 0;C=B-1
	else:A=p.steps;C=int(min(p.denoising_strength,.999)*A)
	return A,C
approximation_indexes={'Full':0,'Approx NN':1,'Approx cheap':2,'TAESD':3}
def samples_to_images_tensor(sample,approximation=_A,model=_A):
	'Transforms 4-channel latent space images into 3-channel RGB image tensors, with values in range [-1, 1].';D=model;C=sample;A=approximation
	if A is _A or shared.state.interrupted and opts.live_preview_fast_interrupt:
		A=approximation_indexes.get(opts.show_progress_type,0);from modules import lowvram as E
		if A==0 and E.is_enabled(shared.sd_model)and not shared.opts.live_preview_allow_lowvram_full:A=1
	if A==2:B=sd_vae_approx.cheap_approximation(C)
	elif A==1:B=sd_vae_approx.model()(C.to(devices.device,devices.dtype)).detach()
	elif A==3:B=sd_vae_taesd.decoder_model()(C.to(devices.device,devices.dtype)).detach();B=B*2-1
	else:
		if D is _A:D=shared.sd_model
		with devices.without_autocast():B=D.decode_first_stage(C.to(D.first_stage_model.dtype))
	return B
def single_sample_to_image(sample,approximation=_A):A=samples_to_images_tensor(sample.unsqueeze(0),approximation)[0]*.5+.5;A=torch.clamp(A,min=.0,max=1.);A=255.*np.moveaxis(A.cpu().numpy(),0,2);A=A.astype(np.uint8);return Image.fromarray(A)
def decode_first_stage(model,x):x=x.to(devices.dtype_vae);A=approximation_indexes.get(opts.sd_vae_decode_method,0);return samples_to_images_tensor(x,A,model)
def sample_to_image(samples,index=0,approximation=_A):return single_sample_to_image(samples[index],approximation)
def samples_to_image_grid(samples,approximation=_A):return images.image_grid([single_sample_to_image(A,approximation)for A in samples])
def images_tensor_to_samples(image,approximation=_A,model=_A):
	'image[0, 1] -> latent';C=approximation;B=model;A=image
	if C is _A:C=approximation_indexes.get(opts.sd_vae_encode_method,0)
	if C==3:A=A.to(devices.device,devices.dtype);D=sd_vae_taesd.encoder_model()(A)
	else:
		if B is _A:B=shared.sd_model
		B.first_stage_model.to(devices.dtype_vae);A=A.to(shared.device,dtype=devices.dtype_vae);A=A*2-1
		if len(A)>1:D=torch.stack([B.get_first_stage_encoding(B.encode_first_stage(torch.unsqueeze(A,0)))[0]for A in A])
		else:D=B.get_first_stage_encoding(B.encode_first_stage(A))
	return D
def store_latent(decoded):
	A=decoded;state.current_latent=A
	if opts.live_previews_enable and opts.show_progress_every_n_steps>0 and shared.state.sampling_step%opts.show_progress_every_n_steps==0:
		if not shared.parallel_processing_allowed:shared.state.assign_current_image(sample_to_image(A))
def is_sampler_using_eta_noise_seed_delta(p):
	'returns whether sampler from config will use eta noise seed delta for image creation';B=sd_samplers.find_sampler_config(p.sampler_name);A=p.eta
	if A is _A and p.sampler is not _A:A=p.sampler.eta
	if A is _A and B is not _A:A=0 if B.options.get('default_eta_is_0',_B)else 1.
	if A==0:return _B
	return B.options.get('uses_ensd',_B)
class InterruptedException(BaseException):0
def replace_torchsde_browinan():
	import torchsde._brownian.brownian_interval
	def A(size,dtype,device,seed):return devices.randn_local(seed,size).to(device=device,dtype=dtype)
	torchsde._brownian.brownian_interval._randn=A
replace_torchsde_browinan()
def apply_refiner(cfg_denoiser):
	E='second pass';A=cfg_denoiser;F=A.step/A.total_steps;C=A.p.refiner_switch_at;B=A.p.refiner_checkpoint_info
	if C is not _A and F<C:return _B
	if B is _A or shared.sd_model.sd_checkpoint_info==B:return _B
	if getattr(A.p,'enable_hr',_B):
		D=A.p.is_hr_pass
		if opts.hires_fix_refiner_pass=='first pass'and D:return _B
		if opts.hires_fix_refiner_pass==E and not D:return _B
		if opts.hires_fix_refiner_pass!=E:A.p.extra_generation_params['Hires refiner']=opts.hires_fix_refiner_pass
	A.p.extra_generation_params['Refiner']=B.short_title;A.p.extra_generation_params['Refiner switch at']=C
	with sd_models.SkipWritingToConfig():sd_models.reload_model_weights(info=B)
	devices.torch_gc();A.p.setup_conds();A.update_inner_model();return True
class TorchHijack:
	'This is here to replace torch.randn_like of k-diffusion.\n\n    k-diffusion has random_sampler argument for most samplers, but not for all, so\n    this is needed to properly replace every use of torch.randn_like.\n\n    We need to replace to make images generated in batches to be same as images generated individually.'
	def __init__(A,p):A.rng=p.rng
	def __getattr__(B,item):
		A=item
		if A=='randn_like':return B.randn_like
		if hasattr(torch,A):return getattr(torch,A)
		raise AttributeError(f"'{type(B).__name__}' object has no attribute '{A}'")
	def randn_like(A,x):return A.rng.next()
class Sampler:
	def __init__(A,funcname):B=funcname;A.funcname=B;A.func=B;A.extra_params=[];A.sampler_noises=_A;A.stop_at=_A;A.eta=_A;A.config=_A;A.last_latent=_A;A.s_min_uncond=_A;A.s_churn=.0;A.s_tmin=.0;A.s_tmax=float('inf');A.s_noise=1.;A.eta_option_field='eta_ancestral';A.eta_infotext_field='Eta';A.eta_default=1.;A.conditioning_key=shared.sd_model.model.conditioning_key;A.p=_A;A.model_wrap_cfg=_A;A.sampler_extra_args=_A;A.options={}
	def callback_state(A,d):
		B=d['i']
		if A.stop_at is not _A and B>A.stop_at:raise InterruptedException
		state.sampling_step=B;shared.total_tqdm.update()
	def launch_sampling(A,steps,func):
		B=steps;A.model_wrap_cfg.steps=B;A.model_wrap_cfg.total_steps=A.config.total_steps(B);state.sampling_steps=B;state.sampling_step=0
		try:return func()
		except RecursionError:print('Encountered RecursionError during sampling, returning last latent. rho >5 with a polyexponential scheduler may cause this error. You should try to use a smaller rho value instead.');return A.last_latent
		except InterruptedException:return A.last_latent
	def number_of_needed_noises(A,p):return p.steps
	def initialize(A,p):
		L='eta';K='s_noise';J='s_tmax';I='s_tmin';H='s_churn';A.p=p;A.model_wrap_cfg.p=p;A.model_wrap_cfg.mask=p.mask if hasattr(p,'mask')else _A;A.model_wrap_cfg.nmask=p.nmask if hasattr(p,'nmask')else _A;A.model_wrap_cfg.step=0;A.model_wrap_cfg.image_cfg_scale=getattr(p,'image_cfg_scale',_A);A.eta=p.eta if p.eta is not _A else getattr(opts,A.eta_option_field,.0);A.s_min_uncond=getattr(p,'s_min_uncond',.0);k_diffusion.sampling.torch=TorchHijack(p);B={}
		for C in A.extra_params:
			if hasattr(p,C)and C in inspect.signature(A.func).parameters:B[C]=getattr(p,C)
		if L in inspect.signature(A.func).parameters:
			if A.eta!=A.eta_default:p.extra_generation_params[A.eta_infotext_field]=A.eta
			B[L]=A.eta
		if len(A.extra_params)>0:
			D=getattr(opts,H,p.s_churn);E=getattr(opts,I,p.s_tmin);F=getattr(opts,J,p.s_tmax)or A.s_tmax;G=getattr(opts,K,p.s_noise)
			if H in B and D!=A.s_churn:B[H]=D;p.s_churn=D;p.extra_generation_params['Sigma churn']=D
			if I in B and E!=A.s_tmin:B[I]=E;p.s_tmin=E;p.extra_generation_params['Sigma tmin']=E
			if J in B and F!=A.s_tmax:B[J]=F;p.s_tmax=F;p.extra_generation_params['Sigma tmax']=F
			if K in B and G!=A.s_noise:B[K]=G;p.s_noise=G;p.extra_generation_params['Sigma noise']=G
		return B
	def create_noise_sampler(F,x,sigmas,p):
		'For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes';A=sigmas
		if shared.opts.no_dpmpp_sde_batch_determinism:return
		from k_diffusion.sampling import BrownianTreeNoiseSampler as B;C,D=A[A>0].min(),A.max();E=p.all_seeds[p.iteration*p.batch_size:(p.iteration+1)*p.batch_size];return B(x,C,D,seed=E)
	def sample(A,p,x,conditioning,unconditional_conditioning,steps=_A,image_conditioning=_A):raise NotImplementedError()
	def sample_img2img(A,p,x,noise,conditioning,unconditional_conditioning,steps=_A,image_conditioning=_A):raise NotImplementedError()