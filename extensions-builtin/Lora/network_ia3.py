_A='weight'
import network
class ModuleTypeIa3(network.ModuleType):
	def create_module(B,net,weights):
		A=weights
		if all(B in A.w for B in[_A]):return NetworkModuleIa3(net,A)
class NetworkModuleIa3(network.NetworkModule):
	def __init__(B,net,weights):A=weights;super().__init__(net,A);B.w=A.w[_A];B.on_input=A.w['on_input'].item()
	def calc_updown(C,orig_weight):
		A=orig_weight;B=C.w.to(A.device,dtype=A.dtype);D=[B.size(0),A.size(1)]
		if C.on_input:D.reverse()
		else:B=B.reshape(-1,1)
		E=A*B;return C.finalize_updown(E,A,D)