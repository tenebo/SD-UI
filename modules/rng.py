_D='mps'
_C='CPU'
_B='NV'
_A=None
import torch
from modules import devices,rng_philox,shared
def randn(seed,shape,generator=_A):
	'Generate a tensor with random numbers from a normal distribution using seed.\n\n    Uses the seed parameter to set the global torch seed; to generate more with that seed, use randn_like/randn_without_seed.';A=generator;B=shape;manual_seed(seed)
	if shared.opts.randn_source==_B:return torch.asarray((A or nv_rng).randn(B),device=devices.device)
	if shared.opts.randn_source==_C or devices.device.type==_D:return torch.randn(B,device=devices.cpu,generator=A).to(devices.device)
	return torch.randn(B,device=devices.device,generator=A)
def randn_local(seed,shape):
	"Generate a tensor with random numbers from a normal distribution using seed.\n\n    Does not change the global random number generator. You can only generate the seed's first tensor using this function.";A=shape
	if shared.opts.randn_source==_B:C=rng_philox.Generator(seed);return torch.asarray(C.randn(A),device=devices.device)
	B=devices.cpu if shared.opts.randn_source==_C or devices.device.type==_D else devices.device;D=torch.Generator(B).manual_seed(int(seed));return torch.randn(A,device=B,generator=D).to(devices.device)
def randn_like(x):
	'Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.\n\n    Use either randn() or manual_seed() to initialize the generator.'
	if shared.opts.randn_source==_B:return torch.asarray(nv_rng.randn(x.shape),device=x.device,dtype=x.dtype)
	if shared.opts.randn_source==_C or x.device.type==_D:return torch.randn_like(x,device=devices.cpu).to(x.device)
	return torch.randn_like(x)
def randn_without_seed(shape,generator=_A):
	'Generate a tensor with random numbers from a normal distribution using the previously initialized genrator.\n\n    Use either randn() or manual_seed() to initialize the generator.';A=generator;B=shape
	if shared.opts.randn_source==_B:return torch.asarray((A or nv_rng).randn(B),device=devices.device)
	if shared.opts.randn_source==_C or devices.device.type==_D:return torch.randn(B,device=devices.cpu,generator=A).to(devices.device)
	return torch.randn(B,device=devices.device,generator=A)
def manual_seed(seed):
	'Set up a global random number generator using the specified seed.'
	if shared.opts.randn_source==_B:global nv_rng;nv_rng=rng_philox.Generator(seed);return
	torch.manual_seed(seed)
def create_generator(seed):
	if shared.opts.randn_source==_B:return rng_philox.Generator(seed)
	A=devices.cpu if shared.opts.randn_source==_C or devices.device.type==_D else devices.device;B=torch.Generator(A).manual_seed(int(seed));return B
def slerp(val,low,high):
	A=high;B=low;C=val;G=B/torch.norm(B,dim=1,keepdim=True);H=A/torch.norm(A,dim=1,keepdim=True);E=(G*H).sum(1)
	if E.mean()>.9995:return B*C+A*(1-C)
	D=torch.acos(E);F=torch.sin(D);I=(torch.sin((1.-C)*D)/F).unsqueeze(1)*B+(torch.sin(C*D)/F).unsqueeze(1)*A;return I
class ImageRNG:
	def __init__(A,shape,seeds,subseeds=_A,subseed_strength=.0,seed_resize_from_h=0,seed_resize_from_w=0):B=seeds;A.shape=tuple(map(int,shape));A.seeds=B;A.subseeds=subseeds;A.subseed_strength=subseed_strength;A.seed_resize_from_h=seed_resize_from_h;A.seed_resize_from_w=seed_resize_from_w;A.generators=[create_generator(A)for A in B];A.is_first=True
	def first(A):
		B=A.shape if A.seed_resize_from_h<=0 or A.seed_resize_from_w<=0 else(A.shape[0],A.seed_resize_from_h//8,A.seed_resize_from_w//8);H=[]
		for(I,(F,J))in enumerate(zip(A.seeds,A.generators)):
			G=_A
			if A.subseeds is not _A and A.subseed_strength!=0:Q=0 if I>=len(A.subseeds)else A.subseeds[I];G=randn(Q,B)
			if B!=A.shape:E=randn(F,B)
			else:E=randn(F,A.shape,generator=J)
			if G is not _A:E=slerp(A.subseed_strength,E,G)
			if B!=A.shape:K=randn(F,A.shape,generator=J);C=(A.shape[2]-B[2])//2;D=(A.shape[1]-B[1])//2;L=B[2]if C>=0 else B[2]+2*C;M=B[1]if D>=0 else B[1]+2*D;N=0 if C<0 else C;O=0 if D<0 else D;C=max(-C,0);D=max(-D,0);K[:,O:O+M,N:N+L]=E[:,D:D+M,C:C+L];E=K
			H.append(E)
		P=shared.opts.eta_noise_seed_delta or 0
		if P:A.generators=[create_generator(A+P)for A in A.seeds]
		return torch.stack(H).to(shared.device)
	def next(A):
		if A.is_first:A.is_first=False;return A.first()
		B=[]
		for C in A.generators:D=randn_without_seed(A.shape,generator=C);B.append(D)
		return torch.stack(B).to(shared.device)
devices.randn=randn
devices.randn_local=randn_local
devices.randn_like=randn_like
devices.randn_without_seed=randn_without_seed
devices.manual_seed=manual_seed