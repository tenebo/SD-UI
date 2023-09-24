_C='random'
_B=False
_A=None
import torch,torch.nn as nn,numpy as np
from einops import rearrange
class VectorQuantizer2(nn.Module):
	'\n    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly\n    avoids costly matrix multiplications and allows for post-hoc remapping of indices.\n    '
	def __init__(A,n_e,e_dim,beta,remap=_A,unknown_index=_C,sane_index_shape=_B,legacy=True):
		super().__init__();A.n_e=n_e;A.e_dim=e_dim;A.beta=beta;A.legacy=legacy;A.embedding=nn.Embedding(A.n_e,A.e_dim);A.embedding.weight.data.uniform_(-1./A.n_e,1./A.n_e);A.remap=remap
		if A.remap is not _A:
			A.register_buffer('used',torch.tensor(np.load(A.remap)));A.re_embed=A.used.shape[0];A.unknown_index=unknown_index
			if A.unknown_index=='extra':A.unknown_index=A.re_embed;A.re_embed=A.re_embed+1
			print(f"Remapping {A.n_e} indices to {A.re_embed} indices. Using {A.unknown_index} for unknown indices.")
		else:A.re_embed=n_e
		A.sane_index_shape=sane_index_shape
	def remap_to_used(C,inds):
		A=inds;D=A.shape;assert len(D)>1;A=A.reshape(D[0],-1);G=C.used.to(A);F=(A[:,:,_A]==G[_A,_A,...]).long();B=F.argmax(-1);E=F.sum(2)<1
		if C.unknown_index==_C:B[E]=torch.randint(0,C.re_embed,size=B[E].shape).to(device=B.device)
		else:B[E]=C.unknown_index
		return B.reshape(D)
	def unmap_to_all(B,inds):
		A=inds;C=A.shape;assert len(C)>1;A=A.reshape(C[0],-1);D=B.used.to(A)
		if B.re_embed>B.used.shape[0]:A[A>=B.used.shape[0]]=0
		E=torch.gather(D[_A,:][A.shape[0]*[0],:],1,A);return E.reshape(C)
	def forward(B,z,temp=_A,rescale_logits=_B,return_logits=_B):
		D='Only for interface compatible with Gumbel';assert temp is _A or temp==1.,D;assert rescale_logits is _B,D;assert return_logits is _B,D;z=rearrange(z,'b c h w -> b h w c').contiguous();E=z.view(-1,B.e_dim);G=torch.sum(E**2,dim=1,keepdim=True)+torch.sum(B.embedding.weight**2,dim=1)-2*torch.einsum('bd,dn->bn',E,rearrange(B.embedding.weight,'n d -> d n'));C=torch.argmin(G,dim=1);A=B.embedding(C).view(z.shape);H=_A;I=_A
		if not B.legacy:F=B.beta*torch.mean((A.detach()-z)**2)+torch.mean((A-z.detach())**2)
		else:F=torch.mean((A.detach()-z)**2)+B.beta*torch.mean((A-z.detach())**2)
		A=z+(A-z).detach();A=rearrange(A,'b h w c -> b c h w').contiguous()
		if B.remap is not _A:C=C.reshape(z.shape[0],-1);C=B.remap_to_used(C);C=C.reshape(-1,1)
		if B.sane_index_shape:C=C.reshape(A.shape[0],A.shape[2],A.shape[3])
		return A,F,(H,I,C)
	def get_codebook_entry(C,indices,shape):
		D=shape;A=indices
		if C.remap is not _A:A=A.reshape(D[0],-1);A=C.unmap_to_all(A);A=A.reshape(-1)
		B=C.embedding(A)
		if D is not _A:B=B.view(D);B=B.permute(0,3,1,2).contiguous()
		return B