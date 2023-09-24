_H='Randomness'
_G='Decode CFG scale'
_F='Decode steps'
_E='sigma_adjustment'
_D='original_negative_prompt'
_C='original_prompt'
_B='c_crossattn'
_A='c_concat'
from collections import namedtuple
import numpy as np
from tqdm import trange
import modules.scripts as scripts,gradio as gr
from modules import processing,shared,sd_samplers,sd_samplers_common
import torch,k_diffusion as K
def find_noise_for_image(p,cond,uncond,cfg_scale,steps):
	I=steps;A=p.init_latent;T=A.new_ones([A.shape[0]])
	if shared.sd_model.parameterization=='v':C=K.external.CompVisVDenoiser(shared.sd_model);J=1
	else:C=K.external.CompVisDenoiser(shared.sd_model);J=0
	B=C.get_sigmas(I).flip(0);shared.state.sampling_steps=I
	for D in trange(1,len(B)):shared.state.sampling_step+=1;E=torch.cat([A]*2);G=torch.cat([B[D]*T]*2);F=torch.cat([uncond,cond]);U=torch.cat([p.image_conditioning]*2);F={_A:[U],_B:[F]};L,M=[K.utils.append_dims(A,E.ndim)for A in C.get_scalings(G)[J:]];N=C.sigma_to_t(G);O=shared.sd_model.apply_model(E*M,N,cond=F);H,P=(E+O*L).chunk(2);Q=H+(P-H)*cfg_scale;R=(A-Q)/B[D];S=B[D]-B[D-1];A=A+R*S;sd_samplers_common.store_latent(A);del E,G,F,L,M,N;del O,H,P,Q,R,S
	shared.state.nextjob();return A/A.std()
Cached=namedtuple('Cached',['noise','cfg_scale','steps','latent',_C,_D,_E])
def find_noise_for_image_sigma_adjustment(p,cond,uncond,cfg_scale,steps):
	M=steps;A=p.init_latent;N=A.new_ones([A.shape[0]])
	if shared.sd_model.parameterization=='v':D=K.external.CompVisVDenoiser(shared.sd_model);O=1
	else:D=K.external.CompVisDenoiser(shared.sd_model);O=0
	B=D.get_sigmas(M).flip(0);shared.state.sampling_steps=M
	for C in trange(1,len(B)):
		shared.state.sampling_step+=1;E=torch.cat([A]*2);G=torch.cat([B[C-1]*N]*2);F=torch.cat([uncond,cond]);U=torch.cat([p.image_conditioning]*2);F={_A:[U],_B:[F]};P,Q=[K.utils.append_dims(A,E.ndim)for A in D.get_scalings(G)[O:]]
		if C==1:H=D.sigma_to_t(torch.cat([B[C]*N]*2))
		else:H=D.sigma_to_t(G)
		R=shared.sd_model.apply_model(E*Q,H,cond=F);I,S=(E+R*P).chunk(2);J=I+(S-I)*cfg_scale
		if C==1:L=(A-J)/(2*B[C])
		else:L=(A-J)/B[C-1]
		T=B[C]-B[C-1];A=A+L*T;sd_samplers_common.store_latent(A);del E,G,F,P,Q,H;del R,I,S,J,L,T
	shared.state.nextjob();return A/B[-1]
class Script(scripts.Script):
	def __init__(A):A.cache=None
	def title(A):return'img2img alternative test'
	def show(A,is_img2img):return is_img2img
	def ui(A,is_img2img):C=.0;B=True;D=gr.Markdown('\n        * `CFG Scale` should be 2 or lower.\n        ');E=gr.Checkbox(label='Override `Sampling method` to Euler?(this method is built for it)',value=B,elem_id=A.elem_id('override_sampler'));F=gr.Checkbox(label='Override `prompt` to the same value as `original prompt`?(and `negative prompt`)',value=B,elem_id=A.elem_id('override_prompt'));G=gr.Textbox(label='Original prompt',lines=1,elem_id=A.elem_id(_C));H=gr.Textbox(label='Original negative prompt',lines=1,elem_id=A.elem_id(_D));I=gr.Checkbox(label='Override `Sampling Steps` to the same value as `Decode steps`?',value=B,elem_id=A.elem_id('override_steps'));J=gr.Slider(label=_F,minimum=1,maximum=150,step=1,value=50,elem_id=A.elem_id('st'));K=gr.Checkbox(label='Override `Denoising strength` to 1?',value=B,elem_id=A.elem_id('override_strength'));L=gr.Slider(label=_G,minimum=C,maximum=15.,step=.1,value=1.,elem_id=A.elem_id('cfg'));M=gr.Slider(label=_H,minimum=C,maximum=1.,step=.01,value=C,elem_id=A.elem_id('randomness'));N=gr.Checkbox(label='Sigma adjustment for finding noise for image',value=False,elem_id=A.elem_id(_E));return[D,E,F,G,H,I,J,K,L,M,N]
	def run(A,p,_,override_sampler,override_prompt,original_prompt,original_negative_prompt,override_steps,st,override_strength,cfg,randomness,sigma_adjustment):
		G=sigma_adjustment;F=randomness;E=cfg;D=original_negative_prompt;C=original_prompt;B=st
		if override_sampler:p.sampler_name='Euler'
		if override_prompt:p.prompt=C;p.negative_prompt=D
		if override_steps:p.steps=B
		if override_strength:p.denoising_strength=1.
		def H(conditioning,unconditional_conditioning,seeds,subseeds,subseed_strength,prompts):
			I=(p.init_latent.cpu().numpy()*10).astype(int);M=A.cache is not None and A.cache.cfg_scale==E and A.cache.steps==B and A.cache.original_prompt==C and A.cache.original_negative_prompt==D and A.cache.sigma_adjustment==G;N=M and A.cache.latent.shape==I.shape and np.abs(A.cache.latent-I).sum()<100
			if N:H=A.cache.noise
			else:
				shared.state.job_count+=1;J=p.sd_model.get_learned_conditioning(p.batch_size*[C]);K=p.sd_model.get_learned_conditioning(p.batch_size*[D])
				if G:H=find_noise_for_image_sigma_adjustment(p,J,K,E,B)
				else:H=find_noise_for_image(p,J,K,E,B)
				A.cache=Cached(H,E,B,I,C,D,G)
			O=processing.create_random_tensors(p.init_latent.shape[1:],seeds=seeds,subseeds=subseeds,subseed_strength=p.subseed_strength,seed_resize_from_h=p.seed_resize_from_h,seed_resize_from_w=p.seed_resize_from_w,p=p);P=((1-F)*H+F*O)/(F**2+(1-F)**2)**.5;L=sd_samplers.create_sampler(p.sampler_name,p.sd_model);Q=L.model_wrap.get_sigmas(p.steps);R=P-p.init_latent/Q[0];p.seed=p.seed+1;return L.sample_img2img(p,p.init_latent,R,conditioning,unconditional_conditioning,image_conditioning=p.image_conditioning)
		p.sample=H;p.extra_generation_params['Decode prompt']=C;p.extra_generation_params['Decode negative prompt']=D;p.extra_generation_params[_G]=E;p.extra_generation_params[_F]=B;p.extra_generation_params[_H]=F;p.extra_generation_params['Sigma Adjustment']=G;I=processing.process_images(p);return I