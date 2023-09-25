_B='b_norm'
_A='w_norm'
import network
class ModuleTypeNorm(network.ModuleType):
	def create_module(B,net,weights):
		A=weights
		if all(B in A.w for B in[_A,_B]):return NetworkModuleNorm(net,A)
class NetworkModuleNorm(network.NetworkModule):
	def __init__(B,net,weights):A=weights;super().__init__(net,A);B.w_norm=A.w.get(_A);B.b_norm=A.w.get(_B)
	def calc_updown(A,orig_weight):
		B=orig_weight;D=A.w_norm.shape;E=A.w_norm.to(B.device,dtype=B.dtype)
		if A.b_norm is not None:C=A.b_norm.to(B.device,dtype=B.dtype)
		else:C=None
		return A.finalize_updown(E,B,D,C)