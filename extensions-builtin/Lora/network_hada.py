_D='hada_w2_b'
_C='hada_w2_a'
_B='hada_w1_b'
_A='hada_w1_a'
import lyco_helpers,network
class ModuleTypeHada(network.ModuleType):
	def create_module(B,net,weights):
		A=weights
		if all(B in A.w for B in[_A,_B,_C,_D]):return NetworkModuleHada(net,A)
class NetworkModuleHada(network.NetworkModule):
	def __init__(A,net,weights):
		B=weights;super().__init__(net,B)
		if hasattr(A.sd_module,'weight'):A.shape=A.sd_module.weight.shape
		A.w1a=B.w[_A];A.w1b=B.w[_B];A.dim=A.w1b.shape[0];A.w2a=B.w[_C];A.w2b=B.w[_D];A.t1=B.w.get('hada_t1');A.t2=B.w.get('hada_t2')
	def calc_updown(B,orig_weight):
		A=orig_weight;E=B.w1a.to(A.device,dtype=A.dtype);C=B.w1b.to(A.device,dtype=A.dtype);F=B.w2a.to(A.device,dtype=A.dtype);G=B.w2b.to(A.device,dtype=A.dtype);D=[E.size(0),C.size(1)]
		if B.t1 is not None:D=[E.size(1),C.size(1)];H=B.t1.to(A.device,dtype=A.dtype);I=lyco_helpers.make_weight_cp(H,E,C);D+=H.shape[2:]
		else:
			if len(C.shape)==4:D+=C.shape[2:]
			I=lyco_helpers.rebuild_conventional(E,C,D)
		if B.t2 is not None:K=B.t2.to(A.device,dtype=A.dtype);J=lyco_helpers.make_weight_cp(K,F,G)
		else:J=lyco_helpers.rebuild_conventional(F,G,D)
		L=I*J;return B.finalize_updown(L,A,D)