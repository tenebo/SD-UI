'SAMPLING ONLY.'
_A=None
import torch
from.uni_pc import NoiseScheduleVP,model_wrapper,UniPC
from modules import shared,devices
class UniPCSampler:
	def __init__(A,model,**D):B=model;super().__init__();A.model=B;C=lambda x:x.clone().detach().to(torch.float32).to(B.device);A.before_sample=_A;A.after_sample=_A;A.register_buffer('alphas_cumprod',C(B.alphas_cumprod))
	def register_buffer(B,name,attr):
		A=attr
		if type(A)==torch.Tensor:
			if A.device!=devices.device:A=A.to(devices.device)
		setattr(B,name,A)
	def set_hooks(A,before_sample,after_sample,after_update):A.before_sample=before_sample;A.after_sample=after_sample;A.after_update=after_update
	@torch.no_grad()
	def sample(self,S,batch_size,shape,conditioning=_A,callback=_A,normals_sequence=_A,img_callback=_A,quantize_x0=False,eta=.0,mask=_A,x0=_A,temperature=1.,noise_dropout=.0,score_corrector=_A,corrector_kwargs=_A,verbose=True,x_T=_A,log_every_t=100,unconditional_guidance_scale=1.,unconditional_conditioning=_A,**Q):
		B=batch_size;C=self;A=conditioning
		if A is not _A:
			if isinstance(A,dict):
				D=A[list(A.keys())[0]]
				while isinstance(D,list):D=D[0]
				E=D.shape[0]
				if E!=B:print(f"Warning: Got {E} conditionings but batch-size is {B}")
			elif isinstance(A,list):
				for D in A:
					if D.shape[0]!=B:print(f"Warning: Got {E} conditionings but batch-size is {B}")
			elif A.shape[0]!=B:print(f"Warning: Got {A.shape[0]} conditionings but batch-size is {B}")
		I,J,K=shape;L=B,I,J,K;F=C.model.betas.device
		if x_T is _A:G=torch.randn(L,device=F)
		else:G=x_T
		H=NoiseScheduleVP('discrete',alphas_cumprod=C.alphas_cumprod);M='v'if C.model.parameterization=='v'else'noise';N=model_wrapper(lambda x,t,c:C.model.apply_model(x,t,c),H,model_type=M,guidance_type='classifier-free',guidance_scale=unconditional_guidance_scale);O=UniPC(N,H,predict_x0=True,thresholding=False,variant=shared.opts.uni_pc_variant,condition=A,unconditional_condition=unconditional_conditioning,before_sample=C.before_sample,after_sample=C.after_sample,after_update=C.after_update);P=O.sample(G,steps=S,skip_type=shared.opts.uni_pc_skip_type,method='multistep',order=shared.opts.uni_pc_order,lower_order_final=shared.opts.uni_pc_lower_order_final);return P.to(F),_A