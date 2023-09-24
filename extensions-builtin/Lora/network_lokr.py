_G='lokr_w2_b'
_F='lokr_w2_a'
_E='lokr_w2'
_D='lokr_w1_b'
_C='lokr_w1_a'
_B='lokr_w1'
_A=None
import torch,lyco_helpers,network
class ModuleTypeLokr(network.ModuleType):
	def create_module(D,net,weights):
		A=weights;B=_B in A.w or _C in A.w and _D in A.w;C=_E in A.w or _F in A.w and _G in A.w
		if B and C:return NetworkModuleLokr(net,A)
def make_kron(orig_shape,w1,w2):
	if len(w2.shape)==4:w1=w1.unsqueeze(2).unsqueeze(2)
	w2=w2.contiguous();return torch.kron(w1,w2).reshape(orig_shape)
class NetworkModuleLokr(network.NetworkModule):
	def __init__(A,net,weights):B=weights;super().__init__(net,B);A.w1=B.w.get(_B);A.w1a=B.w.get(_C);A.w1b=B.w.get(_D);A.dim=A.w1b.shape[0]if A.w1b is not _A else A.dim;A.w2=B.w.get(_E);A.w2a=B.w.get(_F);A.w2b=B.w.get(_G);A.dim=A.w2b.shape[0]if A.w2b is not _A else A.dim;A.t2=B.w.get('lokr_t2')
	def calc_updown(B,orig_weight):
		A=orig_weight
		if B.w1 is not _A:D=B.w1.to(A.device,dtype=A.dtype)
		else:H=B.w1a.to(A.device,dtype=A.dtype);I=B.w1b.to(A.device,dtype=A.dtype);D=H@I
		if B.w2 is not _A:C=B.w2.to(A.device,dtype=A.dtype)
		elif B.t2 is _A:E=B.w2a.to(A.device,dtype=A.dtype);F=B.w2b.to(A.device,dtype=A.dtype);C=E@F
		else:J=B.t2.to(A.device,dtype=A.dtype);E=B.w2a.to(A.device,dtype=A.dtype);F=B.w2b.to(A.device,dtype=A.dtype);C=lyco_helpers.make_weight_cp(J,E,F)
		G=[D.size(0)*C.size(0),D.size(1)*C.size(1)]
		if len(A.shape)==4:G=A.shape
		K=make_kron(G,D,C);return B.finalize_updown(K,A,G)