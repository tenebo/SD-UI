import network
class ModuleTypeFull(network.ModuleType):
	def create_module(B,net,weights):
		A=weights
		if all(B in A.w for B in['diff']):return NetworkModuleFull(net,A)
class NetworkModuleFull(network.NetworkModule):
	def __init__(B,net,weights):A=weights;super().__init__(net,A);B.weight=A.w.get('diff');B.ex_bias=A.w.get('diff_b')
	def calc_updown(A,orig_weight):
		B=orig_weight;D=A.weight.shape;E=A.weight.to(B.device,dtype=B.dtype)
		if A.ex_bias is not None:C=A.ex_bias.to(B.device,dtype=B.dtype)
		else:C=None
		return A.finalize_updown(E,B,D,C)