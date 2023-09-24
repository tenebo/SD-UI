_D='denoised'
_C='sigma_hat'
_B='sigma'
_A=None
import torch,tqdm,k_diffusion.sampling,numpy as np
from modules import shared
from modules.models.diffusion.uni_pc import uni_pc
@torch.no_grad()
def ddim(model,x,timesteps,extra_args=_A,callback=_A,disable=_A,eta=.0):
	H=callback;G=model;C=extra_args;A=timesteps;I=G.inner_model.inner_model.alphas_cumprod;D=I[A];F=I[torch.nn.functional.pad(A[:-1],pad=(1,0))].to(torch.float64 if x.device.type!='mps'else torch.float32);O=torch.sqrt(1-D);P=eta*np.sqrt((1-F.cpu().numpy())/(1-D.cpu())*(1-D.cpu()/F.cpu().numpy()));C={}if C is _A else C;Q=x.new_ones(x.shape[0]);E=x.new_ones((x.shape[0],1,1,1))
	for J in tqdm.trange(len(A)-1,disable=disable):
		B=len(A)-1-J;K=G(x,A[B].item()*Q,**C);R=D[B].item()*E;L=F[B].item()*E;M=P[B].item()*E;S=O[B].item()*E;N=(x-S*K)/R.sqrt();T=(1.-L-M**2).sqrt()*K;U=M*k_diffusion.sampling.torch.randn_like(x);x=L.sqrt()*N+T+U
		if H is not _A:H({'x':x,'i':J,_B:0,_C:0,_D:N})
	return x
@torch.no_grad()
def plms(model,x,timesteps,extra_args=_A,callback=_A,disable=_A):
	J=callback;G=model;D=extra_args;B=timesteps;K=G.inner_model.inner_model.alphas_cumprod;L=K[B];Q=K[torch.nn.functional.pad(B[:-1],pad=(1,0))].to(torch.float64 if x.device.type!='mps'else torch.float32);R=torch.sqrt(1-L);D={}if D is _A else D;M=x.new_ones([x.shape[0]]);H=x.new_ones((x.shape[0],1,1,1));A=[]
	def N(e_t,index):A=index;D=L[A].item()*H;B=Q[A].item()*H;E=R[A].item()*H;C=(x-E*e_t)/D.sqrt();F=(1.-B).sqrt()*e_t;G=B.sqrt()*C+F;return G,C
	for O in tqdm.trange(len(B)-1,disable=disable):
		E=len(B)-1-O;S=B[E].item()*M;T=B[max(E-1,0)].item()*M;C=G(x,S,**D)
		if len(A)==0:I,P=N(C,E);U=G(I,T,**D);F=(C+U)/2
		elif len(A)==1:F=(3*C-A[-1])/2
		elif len(A)==2:F=(23*C-16*A[-1]+5*A[-2])/12
		else:F=(55*C-59*A[-1]+37*A[-2]-9*A[-3])/24
		I,P=N(F,E);A.append(C)
		if len(A)>=4:A.pop(0)
		x=I
		if J is not _A:J({'x':x,'i':O,_B:0,_C:0,_D:P})
	return x
class UniPCCFG(uni_pc.UniPC):
	def __init__(A,cfg_model,extra_args,callback,*C,**D):
		B=callback;super().__init__(_A,*C,**D)
		def E(x,model_x):B({'x':x,'i':A.index,_B:0,_C:0,_D:model_x});A.index+=1
		A.cfg_model=cfg_model;A.extra_args=extra_args;A.callback=B;A.index=0;A.after_update=E
	def get_model_input_time(A,t_continuous):return(t_continuous-1./A.noise_schedule.total_N)*1e3
	def model(A,x,t):B=A.get_model_input_time(t);C=A.cfg_model(x,B,**A.extra_args);return C
def unipc(model,x,timesteps,extra_args=_A,callback=_A,disable=_A,is_img2img=False):B=timesteps;A=model;C=A.inner_model.inner_model.alphas_cumprod;D=uni_pc.NoiseScheduleVP('discrete',alphas_cumprod=C);E=B[-1]/1000+1/1000 if is_img2img else _A;F=UniPCCFG(A,extra_args,callback,D,predict_x0=True,thresholding=False,variant=shared.opts.uni_pc_variant);x=F.sample(x,steps=len(B),t_start=E,skip_type=shared.opts.uni_pc_skip_type,method='multistep',order=shared.opts.uni_pc_order,lower_order_final=shared.opts.uni_pc_lower_order_final);return x