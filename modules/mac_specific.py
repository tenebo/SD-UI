_D='torch.nn.functional.layer_norm'
_C='device'
_B=None
_A='mps'
import logging,torch,platform
from modules.sd_hijack_utils import CondFunc
from packaging import version
from modules import shared
log=logging.getLogger(__name__)
def check_for_mps():
	A=False
	if version.parse(torch.__version__)<=version.parse('2.0.1'):
		if not getattr(torch,'has_mps',A):return A
		try:torch.zeros(1).to(torch.device(_A));return True
		except Exception:return A
	else:return torch.backends.mps.is_available()and torch.backends.mps.is_built()
has_mps=check_for_mps()
def torch_mps_gc():
	try:
		if shared.state.current_latent is not _B:log.debug('`current_latent` is set, skipping MPS garbage collection');return
		from torch.mps import empty_cache as A;A()
	except Exception:log.warning('MPS garbage collection failed',exc_info=True)
def cumsum_fix(input,cumsum_func,*C,**A):
	D=cumsum_func
	if input.device.type==_A:
		B=A.get('dtype',input.dtype)
		if B==torch.int64:return D(input.cpu(),*C,**A).to(input.device)
		elif B==torch.bool or cumsum_needs_int_fix and(B==torch.int8 or B==torch.int16):return D(input.to(torch.int32),*C,**A).to(torch.int64)
	return D(input,*C,**A)
if has_mps:
	if platform.mac_ver()[0].startswith('13.2.'):CondFunc('torch.nn.functional.linear',lambda _,input,weight,bias:torch.matmul(input,weight.t())+bias if bias is not _B else torch.matmul(input,weight.t()),lambda _,input,weight,bias:input.numel()>10485760)
	if version.parse(torch.__version__)<version.parse('1.13'):CondFunc('torch.Tensor.to',lambda orig_func,self,*A,**B:orig_func(self.contiguous(),*A,**B),lambda _,self,*A,**B:self.device.type!=_A and(A and isinstance(A[0],torch.device)and A[0].type==_A or isinstance(B.get(_C),torch.device)and B[_C].type==_A));CondFunc(_D,lambda orig_func,*A,**B:orig_func(*[A[0].contiguous()]+list(A[1:]),**B),lambda _,*A,**B:A and isinstance(A[0],torch.Tensor)and A[0].device.type==_A);CondFunc('torch.Tensor.numpy',lambda orig_func,self,*A,**B:orig_func(self.detach(),*A,**B),lambda _,self,*A,**B:self.requires_grad)
	elif version.parse(torch.__version__)>version.parse('1.13.1'):
		cumsum_needs_int_fix=not torch.Tensor([1,2]).to(torch.device(_A)).equal(torch.ShortTensor([1,1]).to(torch.device(_A)).cumsum(0));cumsum_fix_func=lambda orig_func,input,*A,**B:cumsum_fix(input,orig_func,*A,**B);CondFunc('torch.cumsum',cumsum_fix_func,_B);CondFunc('torch.Tensor.cumsum',cumsum_fix_func,_B);CondFunc('torch.narrow',lambda orig_func,*A,**B:orig_func(*A,**B).clone(),_B);CondFunc(_D,lambda orig_func,x,normalized_shape,weight,bias,eps,**A:orig_func(x.float(),normalized_shape,weight.float()if weight is not _B else _B,bias.float()if bias is not _B else bias,eps).to(x.dtype),lambda _,input,*A,**B:len(A)==4 and input.device.type==_A)
		if platform.processor()=='i386':
			for funcName in['torch.argmax','torch.Tensor.argmax']:CondFunc(funcName,lambda _,input,*A,**B:torch.max(input.float()if input.dtype==torch.int64 else input,*A,**B)[1],lambda _,input,*A,**B:input.device.type==_A)