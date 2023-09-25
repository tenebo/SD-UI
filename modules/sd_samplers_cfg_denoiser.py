_C='uncond'
_B=False
_A=None
import torch
from modules import prompt_parser,devices,sd_samplers_common
from modules.shared import opts,state
import modules.shared as shared
from modules.script_callbacks import CFGDenoiserParams,cfg_denoiser_callback
from modules.script_callbacks import CFGDenoisedParams,cfg_denoised_callback
from modules.script_callbacks import AfterCFGCallbackParams,cfg_after_cfg_callback
def catenate_conds(conds):
	A=conds
	if not isinstance(A[0],dict):return torch.cat(A)
	return{B:torch.cat([A[B]for A in A])for B in A[0].keys()}
def subscript_cond(cond,a,b):
	A=cond
	if not isinstance(A,dict):return A[a:b]
	return{A:B[a:b]for(A,B)in A.items()}
def pad_cond(tensor,repeats,empty):
	D='crossattn';C=empty;B=repeats;A=tensor
	if not isinstance(A,dict):return torch.cat([A,C.repeat((A.shape[0],B,1))],axis=1)
	A[D]=pad_cond(A[D],B,C);return A
class CFGDenoiser(torch.nn.Module):
	'\n    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)\n    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)\n    instead of one. Originally, the second prompt is just an empty string, but we use non-empty\n    negative prompt.\n    '
	def __init__(A,sampler):super().__init__();A.model_wrap=_A;A.mask=_A;A.nmask=_A;A.init_latent=_A;A.steps=_A;'number of steps as specified by user in UI';A.total_steps=_A;'expected number of calls to denoiser calculated from self.steps and specifics of the selected sampler';A.step=0;A.image_cfg_scale=_A;A.padded_cond_uncond=_B;A.sampler=sampler;A.model_wrap=_A;A.p=_A;A.mask_before_denoising=_B
	@property
	def inner_model(self):raise NotImplementedError()
	def combine_denoised(H,x_out,conds_list,uncond,cond_scale):
		A=x_out;B=A[-uncond.shape[0]:];C=torch.clone(B)
		for(D,E)in enumerate(conds_list):
			for(F,G)in E:C[D]+=(A[F]-B[D])*(G*cond_scale)
		return C
	def combine_denoised_for_edit_model(C,x_out,cond_scale):D,A,B=x_out.chunk(3);E=B+cond_scale*(D-A)+C.image_cfg_scale*(A-B);return E
	def get_pred_x0(A,x_in,x_out,sigma):return x_out
	def update_inner_model(A):A.model_wrap=_A;B,C=A.p.get_conds();A.sampler.sampler_extra_args['cond']=B;A.sampler.sampler_extra_args[_C]=C
	def forward(A,x,sigma,uncond,cond,cond_scale,s_min_uncond,image_cond):
		g='c_concat';f='c_crossattn';c=s_min_uncond;b=cond_scale;a=True;T=image_cond;I=sigma;B=uncond
		if state.interrupted or state.skipped:raise sd_samplers_common.InterruptedException
		if sd_samplers_common.apply_refiner(A):cond=A.sampler.sampler_extra_args['cond'];B=A.sampler.sampler_extra_args[_C]
		M=shared.sd_model.cond_stage_key=='edit'and A.image_cfg_scale is not _A and A.image_cfg_scale!=1.;N,D=prompt_parser.reconstruct_multicond_batch(cond,A.step);B=prompt_parser.reconstruct_cond_batch(B,A.step);assert not M or all(len(A)==1 for A in N),'AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)'
		if A.mask_before_denoising and A.mask is not _A:x=A.init_latent*A.mask+A.nmask*x
		H=len(N);O=[len(N[A])for A in range(H)]
		if shared.sd_model.model.conditioning_key=='crossattn-adm':W=torch.zeros_like(T);P=lambda c_crossattn,c_adm:{f:[c_crossattn],'c_adm':c_adm}
		else:
			W=T
			if isinstance(B,dict):P=lambda c_crossattn,c_concat:{**c_crossattn,g:[c_concat]}
			else:P=lambda c_crossattn,c_concat:{f:[c_crossattn],g:[c_concat]}
		if not M:E=torch.cat([torch.stack([x[A]for B in range(B)])for(A,B)in enumerate(O)]+[x]);J=torch.cat([torch.stack([I[A]for B in range(B)])for(A,B)in enumerate(O)]+[I]);L=torch.cat([torch.stack([T[A]for B in range(B)])for(A,B)in enumerate(O)]+[W])
		else:E=torch.cat([torch.stack([x[A]for B in range(B)])for(A,B)in enumerate(O)]+[x]+[x]);J=torch.cat([torch.stack([I[A]for B in range(B)])for(A,B)in enumerate(O)]+[I]+[I]);L=torch.cat([torch.stack([T[A]for B in range(B)])for(A,B)in enumerate(O)]+[W]+[torch.zeros_like(A.init_latent)])
		Q=CFGDenoiserParams(E,L,J,state.sampling_step,state.sampling_steps,D,B);cfg_denoiser_callback(Q);E=Q.x;L=Q.image_cond;J=Q.sigma;D=Q.text_cond;B=Q.text_uncond;R=_B
		if A.step%2 and c>0 and I[0]<c and not M:R=a;E=E[:-H];J=J[:-H]
		A.padded_cond_uncond=_B
		if shared.opts.pad_cond_uncond and D.shape[1]!=B.shape[1]:
			X=shared.sd_model.cond_stage_model_empty_prompt;U=(D.shape[1]-B.shape[1])//X.shape[1]
			if U<0:D=pad_cond(D,-U,X);A.padded_cond_uncond=a
			elif U>0:B=pad_cond(B,U,X);A.padded_cond_uncond=a
		if D.shape[1]==B.shape[1]or R:
			if M:V=catenate_conds([D,B,B])
			elif R:V=D
			else:V=catenate_conds([D,B])
			if shared.opts.batch_cond_uncond:C=A.inner_model(E,J,cond=P(V,L))
			else:
				C=torch.zeros_like(E)
				for Y in range(0,C.shape[0],H):F=Y;G=F+H;C[F:G]=A.inner_model(E[F:G],J[F:G],cond=P(subscript_cond(V,F,G),L[F:G]))
		else:
			C=torch.zeros_like(E);H=H*2 if shared.opts.batch_cond_uncond else H
			for Y in range(0,D.shape[0],H):
				F=Y;G=min(F+H,D.shape[0])
				if not M:d=subscript_cond(D,F,G)
				else:d=torch.cat([D[F:G]],B)
				C[F:G]=A.inner_model(E[F:G],J[F:G],cond=P(d,L[F:G]))
			if not R:C[-B.shape[0]:]=A.inner_model(E[-B.shape[0]:],J[-B.shape[0]:],cond=P(B,L[-B.shape[0]:]))
		S=[A[0][0]for A in N]
		if R:h=torch.cat([C[A:A+1]for A in S]);C=torch.cat([C,h])
		i=CFGDenoisedParams(C,state.sampling_step,state.sampling_steps,A.inner_model);cfg_denoised_callback(i);devices.test_for_nans(C,'unet')
		if M:K=A.combine_denoised_for_edit_model(C,b)
		elif R:K=A.combine_denoised(C,N,B,1.)
		else:K=A.combine_denoised(C,N,B,b)
		if not A.mask_before_denoising and A.mask is not _A:K=A.init_latent*A.mask+A.nmask*K
		A.sampler.last_latent=A.get_pred_x0(torch.cat([E[A:A+1]for A in S]),torch.cat([C[A:A+1]for A in S]),I)
		if opts.live_preview_content=='Prompt':Z=A.sampler.last_latent
		elif opts.live_preview_content=='Negative prompt':Z=A.get_pred_x0(E[-B.shape[0]:],C[-B.shape[0]:],I)
		else:Z=A.get_pred_x0(torch.cat([E[A:A+1]for A in S]),torch.cat([K[A:A+1]for A in S]),I)
		sd_samplers_common.store_latent(Z);e=AfterCFGCallbackParams(K,state.sampling_step,state.sampling_steps);cfg_after_cfg_callback(e);K=e.x;A.step+=1;return K