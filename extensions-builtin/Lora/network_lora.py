_D='lora_mid.weight'
_C='lora_up.weight'
_B='lora_down.weight'
_A=False
import torch,lyco_helpers,network
from modules import devices
class ModuleTypeLora(network.ModuleType):
	def create_module(B,net,weights):
		A=weights
		if all(B in A.w for B in[_C,_B]):return NetworkModuleLora(net,A)
class NetworkModuleLora(network.NetworkModule):
	def __init__(A,net,weights):B=weights;super().__init__(net,B);A.up_model=A.create_module(B.w,_C);A.down_model=A.create_module(B.w,_B);A.mid_model=A.create_module(B.w,_D,none_ok=True);A.dim=B.w[_B].shape[0]
	def create_module(B,weights,key,none_ok=_A):
		D=key;A=weights.get(D)
		if A is None and none_ok:return
		F=type(B.sd_module)in[torch.nn.Linear,torch.nn.modules.linear.NonDynamicallyQuantizableLinear,torch.nn.MultiheadAttention];E=type(B.sd_module)in[torch.nn.Conv2d]
		if F:A=A.reshape(A.shape[0],-1);C=torch.nn.Linear(A.shape[1],A.shape[0],bias=_A)
		elif E and D==_B or D=='dyn_up':
			if len(A.shape)==2:A=A.reshape(A.shape[0],-1,1,1)
			if A.shape[2]!=1 or A.shape[3]!=1:C=torch.nn.Conv2d(A.shape[1],A.shape[0],B.sd_module.kernel_size,B.sd_module.stride,B.sd_module.padding,bias=_A)
			else:C=torch.nn.Conv2d(A.shape[1],A.shape[0],(1,1),bias=_A)
		elif E and D==_D:C=torch.nn.Conv2d(A.shape[1],A.shape[0],B.sd_module.kernel_size,B.sd_module.stride,B.sd_module.padding,bias=_A)
		elif E and D==_C or D=='dyn_down':C=torch.nn.Conv2d(A.shape[1],A.shape[0],(1,1),bias=_A)
		else:raise AssertionError(f"Lora layer {B.network_key} matched a layer with unsupported type: {type(B.sd_module).__name__}")
		with torch.no_grad():
			if A.shape!=C.weight.shape:A=A.reshape(C.weight.shape)
			C.weight.copy_(A)
		C.to(device=devices.cpu,dtype=devices.dtype);C.weight.requires_grad_(_A);return C
	def calc_updown(B,orig_weight):
		A=orig_weight;E=B.up_model.weight.to(A.device,dtype=A.dtype);C=B.down_model.weight.to(A.device,dtype=A.dtype);D=[E.size(0),C.size(1)]
		if B.mid_model is not None:F=B.mid_model.weight.to(A.device,dtype=A.dtype);G=lyco_helpers.rebuild_cp_decomposition(E,C,F);D+=F.shape[2:]
		else:
			if len(C.shape)==4:D+=C.shape[2:]
			G=lyco_helpers.rebuild_conventional(E,C,D,B.network.dyn_dim)
		return B.finalize_updown(G,A,D)
	def forward(A,x,y):A.up_model.to(device=devices.device);A.down_model.to(device=devices.device);return y+A.up_model(A.down_model(x))*A.multiplier()*A.calc_scale()