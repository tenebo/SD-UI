from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional,NamedTuple,List
def narrow_trunc(input,dim,start,length):C=length;B=start;A=dim;return torch.narrow(input,A,B,C if input.shape[A]>=B+C else input.shape[A]-B)
class AttnChunk(NamedTuple):exp_values:Tensor;exp_weights_sum:Tensor;max_score:Tensor
class SummarizeChunk:
	@staticmethod
	def __call__(query,key,value):...
class ComputeQueryChunkAttn:
	@staticmethod
	def __call__(query,key,value):...
def _summarize_chunk(query,key,value,scale):D=value;B=query;E=torch.baddbmm(torch.zeros(1,1,1,device=B.device,dtype=B.dtype),B,key.transpose(1,2),alpha=scale,beta=0);A,G=torch.max(E,-1,keepdim=True);A=A.detach();C=torch.exp(E-A);F=torch.bmm(C,D)if B.device.type=='mps'else torch.bmm(C,D.to(C.dtype)).to(D.dtype);A=A.squeeze(-1);return AttnChunk(F,C.sum(dim=-1),A)
def _query_chunk_attention(query,key,value,summarize_chunk,kv_chunk_size):
	B=value;A=kv_chunk_size;O,H,P=key.shape;C,C,Q=B.shape
	def I(chunk_idx):C=chunk_idx;D=narrow_trunc(key,1,C,A);E=narrow_trunc(B,1,C,A);return summarize_chunk(query,D,E)
	J=[I(A)for A in torch.arange(0,H,A)];K=AttnChunk(*map(torch.stack,zip(*J)));D,E,F=K;L,C=torch.max(F,0,keepdim=True);G=torch.exp(F-L);D*=torch.unsqueeze(G,-1);E*=G;M=D.sum(dim=0);N=torch.unsqueeze(E,-1).sum(dim=0);return M/N
def _get_attention_scores_no_kv_chunking(query,key,value,scale):B=value;A=query;D=torch.baddbmm(torch.zeros(1,1,1,device=A.device,dtype=A.dtype),A,key.transpose(1,2),alpha=scale,beta=0);C=D.softmax(dim=-1);del D;E=torch.bmm(C,B)if A.device.type=='mps'else torch.bmm(C,B.to(C.dtype)).to(B.dtype);return E
class ScannedChunk(NamedTuple):chunk_idx:int;attn_chunk:AttnChunk
def efficient_dot_product_attention(query,key,value,query_chunk_size=1024,kv_chunk_size=None,kv_chunk_size_min=None,use_checkpoint=True):
	"Computes efficient dot-product attention given query, key, and value.\n      This is efficient version of attention presented in\n      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.\n      Args:\n        query: queries for calculating attention with shape of\n          `[batch * num_heads, tokens, channels_per_head]`.\n        key: keys for calculating attention with shape of\n          `[batch * num_heads, tokens, channels_per_head]`.\n        value: values to be used in attention with shape of\n          `[batch * num_heads, tokens, channels_per_head]`.\n        query_chunk_size: int: query chunks size\n        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)\n        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).\n        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)\n      Returns:\n        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.\n      ";J=kv_chunk_size_min;I=value;E=key;C=query;B=kv_chunk_size;A=query_chunk_size;R,F,O=C.shape;P,G,P=E.shape;K=O**-.5;B=min(B or int(math.sqrt(G)),G)
	if J is not None:B=max(B,J)
	def Q(chunk_idx):return narrow_trunc(C,1,chunk_idx,min(A,F))
	D=partial(_summarize_chunk,scale=K);D=partial(checkpoint,D)if use_checkpoint else D;L=partial(_get_attention_scores_no_kv_chunking,scale=K)if G<=B else partial(_query_chunk_attention,kv_chunk_size=B,summarize_chunk=D)
	if F<=A:return L(query=C,key=E,value=I)
	M=torch.zeros_like(C)
	for H in range(math.ceil(F/A)):N=L(query=Q(H*A),key=E,value=I);M[:,H*A:H*A+N.shape[1],:]=N
	return M