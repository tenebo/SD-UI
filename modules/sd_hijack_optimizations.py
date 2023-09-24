from __future__ import annotations
_M='reserved_bytes.all.current'
_L='active_bytes.all.current'
_K='xformers'
_J='b (h w) c -> b c h w'
_I='b c h w -> b (h w) c'
_H='(b h) n d -> b n (h d)'
_G='b i j, b j d -> b i d'
_F='b i d, b j d -> b i j'
_E='b n (h d) -> (b h) n d'
_D=False
_C='mps'
_B=True
_A=None
import math,psutil,platform,torch
from torch import einsum
from ldm.util import default
from einops import rearrange
from modules import shared,errors,devices,sub_quadratic_attention
from modules.hypernetworks import hypernetwork
import ldm.modules.attention,ldm.modules.diffusionmodules.model,sgm.modules.attention,sgm.modules.diffusionmodules.model
diffusionmodules_model_AttnBlock_forward=ldm.modules.diffusionmodules.model.AttnBlock.forward
sgm_diffusionmodules_model_AttnBlock_forward=sgm.modules.diffusionmodules.model.AttnBlock.forward
class SdOptimization:
	name=_A;label=_A;cmd_opt=_A;priority=0
	def title(A):
		if A.label is _A:return A.name
		return f"{A.name} - {A.label}"
	def is_available(A):return _B
	def apply(A):0
	def undo(A):ldm.modules.attention.CrossAttention.forward=hypernetwork.attention_CrossAttention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=diffusionmodules_model_AttnBlock_forward;sgm.modules.attention.CrossAttention.forward=hypernetwork.attention_CrossAttention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=sgm_diffusionmodules_model_AttnBlock_forward
class SdOptimizationXformers(SdOptimization):
	name=_K;cmd_opt=_K;priority=100
	def is_available(A):return shared.cmd_opts.force_enable_xformers or shared.xformers_available and torch.cuda.is_available()and(6,0)<=torch.cuda.get_device_capability(shared.device)<=(9,0)
	def apply(A):ldm.modules.attention.CrossAttention.forward=xformers_attention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=xformers_attnblock_forward;sgm.modules.attention.CrossAttention.forward=xformers_attention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=xformers_attnblock_forward
class SdOptimizationSdpNoMem(SdOptimization):
	name='sdp-no-mem';label='scaled dot product without memory efficient attention';cmd_opt='opt_sdp_no_mem_attention';priority=80
	def is_available(A):return hasattr(torch.nn.functional,'scaled_dot_product_attention')and callable(torch.nn.functional.scaled_dot_product_attention)
	def apply(A):ldm.modules.attention.CrossAttention.forward=scaled_dot_product_no_mem_attention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=sdp_no_mem_attnblock_forward;sgm.modules.attention.CrossAttention.forward=scaled_dot_product_no_mem_attention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=sdp_no_mem_attnblock_forward
class SdOptimizationSdp(SdOptimizationSdpNoMem):
	name='sdp';label='scaled dot product';cmd_opt='opt_sdp_attention';priority=70
	def apply(A):ldm.modules.attention.CrossAttention.forward=scaled_dot_product_attention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=sdp_attnblock_forward;sgm.modules.attention.CrossAttention.forward=scaled_dot_product_attention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=sdp_attnblock_forward
class SdOptimizationSubQuad(SdOptimization):
	name='sub-quadratic';cmd_opt='opt_sub_quad_attention'
	@property
	def priority(self):return 1000 if shared.device.type==_C else 10
	def apply(A):ldm.modules.attention.CrossAttention.forward=sub_quad_attention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=sub_quad_attnblock_forward;sgm.modules.attention.CrossAttention.forward=sub_quad_attention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=sub_quad_attnblock_forward
class SdOptimizationV1(SdOptimization):
	name='V1';label='original v1';cmd_opt='opt_split_attention_v1';priority=10
	def apply(A):ldm.modules.attention.CrossAttention.forward=split_cross_attention_forward_v1;sgm.modules.attention.CrossAttention.forward=split_cross_attention_forward_v1
class SdOptimizationInvokeAI(SdOptimization):
	name='InvokeAI';cmd_opt='opt_split_attention_invokeai'
	@property
	def priority(self):return 1000 if shared.device.type!=_C and not torch.cuda.is_available()else 10
	def apply(A):ldm.modules.attention.CrossAttention.forward=split_cross_attention_forward_invokeAI;sgm.modules.attention.CrossAttention.forward=split_cross_attention_forward_invokeAI
class SdOptimizationDoggettx(SdOptimization):
	name='Doggettx';cmd_opt='opt_split_attention';priority=90
	def apply(A):ldm.modules.attention.CrossAttention.forward=split_cross_attention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=cross_attention_attnblock_forward;sgm.modules.attention.CrossAttention.forward=split_cross_attention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=cross_attention_attnblock_forward
def list_optimizers(res):res.extend([SdOptimizationXformers(),SdOptimizationSdpNoMem(),SdOptimizationSdp(),SdOptimizationSubQuad(),SdOptimizationV1(),SdOptimizationInvokeAI(),SdOptimizationDoggettx()])
if shared.cmd_opts.xformers or shared.cmd_opts.force_enable_xformers:
	try:import xformers.ops;shared.xformers_available=_B
	except Exception:errors.report('Cannot import xformers',exc_info=_B)
def get_available_vram():
	if shared.device.type=='cuda':A=torch.cuda.memory_stats(shared.device);B=A[_L];C=A[_M];D,G=torch.cuda.mem_get_info(torch.cuda.current_device());E=C-B;F=D+E;return F
	else:return psutil.virtual_memory().available
def split_cross_attention_forward_v1(self,x,context=_A,mask=_A,**S):
	F=context;B=self;J=B.heads;K=B.to_q(x);F=default(F,x);L,M=hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks,F);N=B.to_k(L);O=B.to_v(M);del F,L,M,x;A,G,C=(rearrange(A,_E,h=J)for A in(K,N,O));del K,N,O;Q=A.dtype
	if shared.opts.upcast_attn:A,G,C=A.float(),G.float(),C.float()
	with devices.without_autocast(disable=not shared.opts.upcast_attn):
		D=torch.zeros(A.shape[0],A.shape[1],C.shape[2],device=A.device,dtype=A.dtype)
		for E in range(0,A.shape[0],2):H=E+2;I=einsum(_F,A[E:H],G[E:H]);I*=B.scale;P=I.softmax(dim=-1);del I;D[E:H]=einsum(_G,P,C[E:H]);del P
		del A,G,C
	D=D.to(Q);R=rearrange(D,_H,h=J);del D;return B.to_out(R)
def split_cross_attention_forward(self,x,context=_A,mask=_A,**a):
	G=context;B=self;N=B.heads;E=B.to_q(x);G=default(G,x);U,V=hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks,G);C=B.to_k(U);D=B.to_v(V);W=E.dtype
	if shared.opts.upcast_attn:E,C,D=E.float(),C.float(),D if D.device.type==_C else D.float()
	with devices.without_autocast(disable=not shared.opts.upcast_attn):
		C=C*B.scale;del G,x;A,I,J=(rearrange(A,_E,h=N)for A in(E,C,D));del E,C,D;F=torch.zeros(A.shape[0],A.shape[1],J.shape[2],device=A.device,dtype=A.dtype);H=get_available_vram();O=1024**3;X=A.shape[0]*A.shape[1]*I.shape[1]*A.element_size();Y=3 if A.element_size()==2 else 2.5;K=X*Y;L=1
		if K>H:L=2**math.ceil(math.log(K/H,2))
		if L>64:P=math.floor(math.sqrt(math.sqrt(H/2.5))/8)*64;raise RuntimeError(f"Not enough memory, use lower resolution (max approx. {P}x{P}). Need: {K/64/O:0.1f}GB free, Have:{H/O:0.1f}GB free")
		Q=A.shape[1]//L
		for M in range(0,A.shape[1],Q):R=min(M+Q,A.shape[1]);S=einsum(_F,A[:,M:R],I);T=S.softmax(dim=-1,dtype=A.dtype);del S;F[:,M:R]=einsum(_G,T,J);del T
		del A,I,J
	F=F.to(W);Z=rearrange(F,_H,h=N);del F;return B.to_out(Z)
mem_total_gb=psutil.virtual_memory().total//(1<<30)
def einsum_op_compvis(q,k,v):A=einsum(_F,q,k);A=A.softmax(dim=-1,dtype=A.dtype);return einsum(_G,A,v)
def einsum_op_slice_0(q,k,v,slice_size):
	C=slice_size;D=torch.zeros(q.shape[0],q.shape[1],v.shape[2],device=q.device,dtype=q.dtype)
	for A in range(0,q.shape[0],C):B=A+C;D[A:B]=einsum_op_compvis(q[A:B],k[A:B],v[A:B])
	return D
def einsum_op_slice_1(q,k,v,slice_size):
	B=slice_size;C=torch.zeros(q.shape[0],q.shape[1],v.shape[2],device=q.device,dtype=q.dtype)
	for A in range(0,q.shape[1],B):D=A+B;C[:,A:D]=einsum_op_compvis(q[:,A:D],k,v)
	return C
def einsum_op_mps_v1(q,k,v):
	if q.shape[0]*q.shape[1]<=2**16:return einsum_op_compvis(q,k,v)
	else:
		A=math.floor(2**30/(q.shape[0]*q.shape[1]))
		if A%4096==0:A-=1
		return einsum_op_slice_1(q,k,v,A)
def einsum_op_mps_v2(q,k,v):
	if mem_total_gb>8 and q.shape[0]*q.shape[1]<=2**16:return einsum_op_compvis(q,k,v)
	else:return einsum_op_slice_0(q,k,v,1)
def einsum_op_tensor_mem(q,k,v,max_tensor_mb):
	B=max_tensor_mb;C=q.shape[0]*q.shape[1]*k.shape[1]*q.element_size()//(1<<20)
	if C<=B:return einsum_op_compvis(q,k,v)
	A=1<<int((C-1)/B).bit_length()
	if A<=q.shape[0]:return einsum_op_slice_0(q,k,v,q.shape[0]//A)
	return einsum_op_slice_1(q,k,v,max(q.shape[1]//A,1))
def einsum_op_cuda(q,k,v):A=torch.cuda.memory_stats(q.device);B=A[_L];C=A[_M];D,G=torch.cuda.mem_get_info(q.device);E=C-B;F=D+E;return einsum_op_tensor_mem(q,k,v,F/3.3/(1<<20))
def einsum_op(q,k,v):
	if q.device.type=='cuda':return einsum_op_cuda(q,k,v)
	if q.device.type==_C:
		if mem_total_gb>=32 and q.shape[0]%32!=0 and q.shape[0]*q.shape[1]<2**18:return einsum_op_mps_v1(q,k,v)
		return einsum_op_mps_v2(q,k,v)
	return einsum_op_tensor_mem(q,k,v,32)
def split_cross_attention_forward_invokeAI(self,x,context=_A,mask=_A,**K):
	E=context;C=self;G=C.heads;D=C.to_q(x);E=default(E,x);H,I=hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks,E);A=C.to_k(H);B=C.to_v(I);del E,H,I,x;J=D.dtype
	if shared.opts.upcast_attn:D,A,B=D.float(),A.float(),B if B.device.type==_C else B.float()
	with devices.without_autocast(disable=not shared.opts.upcast_attn):A=A*C.scale;D,A,B=(rearrange(A,_E,h=G)for A in(D,A,B));F=einsum_op(D,A,B)
	F=F.to(J);return C.to_out(rearrange(F,_H,h=G))
def sub_quad_attention_forward(self,x,context=_A,mask=_A,**L):
	E=context;C=self;assert mask is _A,'attention-mask not currently implemented for SubQuadraticCrossAttnProcessor.';F=C.heads;A=C.to_q(x);E=default(E,x);G,H=hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks,E);B=C.to_k(G);D=C.to_v(H);del E,G,H,x;A=A.unflatten(-1,(F,-1)).transpose(1,2).flatten(end_dim=1);B=B.unflatten(-1,(F,-1)).transpose(1,2).flatten(end_dim=1);D=D.unflatten(-1,(F,-1)).transpose(1,2).flatten(end_dim=1)
	if A.device.type==_C:A,B,D=A.contiguous(),B.contiguous(),D.contiguous()
	I=A.dtype
	if shared.opts.upcast_attn:A,B=A.float(),B.float()
	x=sub_quad_attention(A,B,D,q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold,use_checkpoint=C.training);x=x.to(I);x=x.unflatten(0,(-1,F)).transpose(1,2).flatten(start_dim=2);J,K=C.to_out;x=J(x);x=K(x);return x
def sub_quad_attention(q,k,v,q_chunk_size=1024,kv_chunk_size=_A,kv_chunk_size_min=_A,chunk_threshold=_A,use_checkpoint=_B):
	E=kv_chunk_size;C=chunk_threshold;B=kv_chunk_size_min;D=torch.finfo(q.dtype).bits//8;F,I,G=q.shape;G,H,G=k.shape;J=F*D*I*H
	if C is _A:
		if q.device.type==_C:A=268435456*(2 if platform.processor()=='i386'else D)
		else:A=int(get_available_vram()*.7)
	elif C==0:A=_A
	else:A=int(.01*C*get_available_vram())
	if B is _A and A is not _A:B=A//(F*D*(k.shape[2]+v.shape[2]))
	elif B==0:B=_A
	if A is not _A and J<=A:E=H
	with devices.without_autocast(disable=q.dtype==v.dtype):return sub_quadratic_attention.efficient_dot_product_attention(q,k,v,query_chunk_size=q_chunk_size,kv_chunk_size=E,kv_chunk_size_min=B,use_checkpoint=use_checkpoint)
def get_xformers_flash_attention_op(q,k,v):
	if not shared.cmd_opts.xformers_flash_attention:return
	try:
		A=xformers.ops.MemoryEfficientAttentionFlashAttentionOp;B,D=A
		if B.supports(xformers.ops.fmha.Inputs(query=q,key=k,value=v,attn_bias=_A)):return A
	except Exception as C:errors.display_once(C,'enabling flash attention')
def xformers_attention_forward(self,x,context=_A,mask=_A,**N):
	F=context;A=self;G=A.heads;H=A.to_q(x);F=default(F,x);K,L=hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks,F);I=A.to_k(K);J=A.to_v(L);B,D,E=(rearrange(A,'b n (h d) -> b n h d',h=G)for A in(H,I,J));del H,I,J;M=B.dtype
	if shared.opts.upcast_attn:B,D,E=B.float(),D.float(),E.float()
	C=xformers.ops.memory_efficient_attention(B,D,E,attn_bias=_A,op=get_xformers_flash_attention_op(B,D,E));C=C.to(M);C=rearrange(C,'b n h d -> b n (h d)',h=G);return A.to_out(C)
def scaled_dot_product_attention_forward(self,x,context=_A,mask=_A,**S):
	H=context;C=mask;B=self;D,N,O=x.shape
	if C is not _A:C=B.prepare_attention_mask(C,N,D);C=C.view(D,B.heads,-1,C.shape[-1])
	E=B.heads;K=B.to_q(x);H=default(H,x);P,Q=hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks,H);L=B.to_k(P);M=B.to_v(Q);F=O//E;G=K.view(D,-1,E,F).transpose(1,2);I=L.view(D,-1,E,F).transpose(1,2);J=M.view(D,-1,E,F).transpose(1,2);del K,L,M;R=G.dtype
	if shared.opts.upcast_attn:G,I,J=G.float(),I.float(),J.float()
	A=torch.nn.functional.scaled_dot_product_attention(G,I,J,attn_mask=C,dropout_p=.0,is_causal=_D);A=A.transpose(1,2).reshape(D,-1,E*F);A=A.to(R);A=B.to_out[0](A);A=B.to_out[1](A);return A
def scaled_dot_product_no_mem_attention_forward(self,x,context=_A,mask=_A,**A):
	with torch.backends.cuda.sdp_kernel(enable_flash=_B,enable_math=_B,enable_mem_efficient=_D):return scaled_dot_product_attention_forward(self,x,context,mask)
def cross_attention_attnblock_forward(self,x):
	C=self;B=x;B=C.norm(B);H=C.q(B);L=C.k(B);Y=C.v(B);E,D,F,G=H.shape;M=H.reshape(E,D,F*G);del H;A=M.permute(0,2,1);del M;I=L.reshape(E,D,F*G);del L;B=torch.zeros_like(I,device=A.device);N=get_available_vram();Z=A.shape[0]*A.shape[1]*I.shape[2]*A.element_size();O=Z*2.5;J=1
	if O>N:J=2**math.ceil(math.log(O/N,2))
	P=A.shape[1]//J if A.shape[1]%J==0 else A.shape[1]
	for K in range(0,A.shape[1],P):Q=K+P;R=torch.bmm(A[:,K:Q],I);S=R*int(D)**-.5;del R;T=torch.nn.functional.softmax(S,dim=2,dtype=A.dtype);del S;U=Y.reshape(E,D,F*G);V=T.permute(0,2,1);del T;B[:,:,K:Q]=torch.bmm(U,V);del U,V
	W=B.reshape(E,D,F,G);del B;X=C.proj_out(W);del W;X+=x;return X
def xformers_attnblock_forward(self,x):
	D=self
	try:
		F=x;F=D.norm(F);A=D.q(F);B=D.k(F);E=D.v(F);I,J,G,K=A.shape;A,B,E=(rearrange(A,_I)for A in(A,B,E));H=A.dtype
		if shared.opts.upcast_attn:A,B=A.float(),B.float()
		A=A.contiguous();B=B.contiguous();E=E.contiguous();C=xformers.ops.memory_efficient_attention(A,B,E,op=get_xformers_flash_attention_op(A,B,E));C=C.to(H);C=rearrange(C,_J,h=G);C=D.proj_out(C);return x+C
	except NotImplementedError:return cross_attention_attnblock_forward(D,x)
def sdp_attnblock_forward(self,x):
	E=self;F=x;F=E.norm(F);A=E.q(F);B=E.k(F);C=E.v(F);I,J,G,K=A.shape;A,B,C=(rearrange(A,_I)for A in(A,B,C));H=A.dtype
	if shared.opts.upcast_attn:A,B,C=A.float(),B.float(),C.float()
	A=A.contiguous();B=B.contiguous();C=C.contiguous();D=torch.nn.functional.scaled_dot_product_attention(A,B,C,dropout_p=.0,is_causal=_D);D=D.to(H);D=rearrange(D,_J,h=G);D=E.proj_out(D);return x+D
def sdp_no_mem_attnblock_forward(self,x):
	with torch.backends.cuda.sdp_kernel(enable_flash=_B,enable_math=_B,enable_mem_efficient=_D):return sdp_attnblock_forward(self,x)
def sub_quad_attnblock_forward(self,x):A=self;C=x;C=A.norm(C);B=A.q(C);D=A.k(C);E=A.v(C);H,I,G,J=B.shape;B,D,E=(rearrange(A,_I)for A in(B,D,E));B=B.contiguous();D=D.contiguous();E=E.contiguous();F=sub_quad_attention(B,D,E,q_chunk_size=shared.cmd_opts.sub_quad_q_chunk_size,kv_chunk_size=shared.cmd_opts.sub_quad_kv_chunk_size,chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold,use_checkpoint=A.training);F=rearrange(F,_J,h=G);F=A.proj_out(F);return x+F