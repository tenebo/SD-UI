'\nVQGAN code, adapted from the original created by the Unleashing Transformers authors:\nhttps://github.com/samb-t/unleashing-transformers/blob/master/models/vqgan.py\n\n'
_H='Wrong params!'
_G='min_encoding_indices'
_F='nearest'
_E='params'
_D=False
_C='cpu'
_B=True
_A=None
import torch,torch.nn as nn,torch.nn.functional as F
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY
def normalize(in_channels):return torch.nn.GroupNorm(num_groups=32,num_channels=in_channels,eps=1e-06,affine=_B)
@torch.jit.script
def swish(x):return x*torch.sigmoid(x)
class VectorQuantizer(nn.Module):
	def __init__(A,codebook_size,emb_dim,beta):super(VectorQuantizer,A).__init__();A.codebook_size=codebook_size;A.emb_dim=emb_dim;A.beta=beta;A.embedding=nn.Embedding(A.codebook_size,A.emb_dim);A.embedding.weight.data.uniform_(-1./A.codebook_size,1./A.codebook_size)
	def forward(B,z):z=z.permute(0,2,3,1).contiguous();F=z.view(-1,B.emb_dim);G=(F**2).sum(dim=1,keepdim=_B)+(B.embedding.weight**2).sum(1)-2*torch.matmul(F,B.embedding.weight.t());I=torch.mean(G);D,E=torch.topk(G,1,dim=1,largest=_D);D=torch.exp(-D/10);C=torch.zeros(E.shape[0],B.codebook_size).to(z);C.scatter_(1,E,1);A=torch.matmul(C,B.embedding.weight).view(z.shape);J=torch.mean((A.detach()-z)**2)+B.beta*torch.mean((A-z.detach())**2);A=z+(A-z).detach();H=torch.mean(C,dim=0);K=torch.exp(-torch.sum(H*torch.log(H+1e-10)));A=A.permute(0,3,1,2).contiguous();return A,J,{'perplexity':K,'min_encodings':C,_G:E,'min_encoding_scores':D,'mean_distance':I}
	def get_codebook_feat(C,indices,shape):
		D=shape;A=indices;A=A.view(-1,1);E=torch.zeros(A.shape[0],C.codebook_size).to(A);E.scatter_(1,A,1);B=torch.matmul(E.float(),C.embedding.weight)
		if D is not _A:B=B.view(D).permute(0,3,1,2).contiguous()
		return B
class GumbelQuantizer(nn.Module):
	def __init__(A,codebook_size,emb_dim,num_hiddens,straight_through=_D,kl_weight=.0005,temp_init=1.):C=emb_dim;B=codebook_size;super().__init__();A.codebook_size=B;A.emb_dim=C;A.straight_through=straight_through;A.temperature=temp_init;A.kl_weight=kl_weight;A.proj=nn.Conv2d(num_hiddens,B,1);A.embed=nn.Embedding(B,C)
	def forward(A,z):E=A.straight_through if A.training else _B;B=A.proj(z);C=F.gumbel_softmax(B,tau=A.temperature,dim=1,hard=E);G=torch.einsum('b n h w, n d -> b d h w',C,A.embed.weight);D=F.softmax(B,dim=1);H=A.kl_weight*torch.sum(D*torch.log(D*A.codebook_size+1e-10),dim=1).mean();I=C.argmax(dim=1);return G,H,{_G:I}
class Downsample(nn.Module):
	def __init__(B,in_channels):A=in_channels;super().__init__();B.conv=torch.nn.Conv2d(A,A,kernel_size=3,stride=2,padding=0)
	def forward(A,x):B=0,1,0,1;x=torch.nn.functional.pad(x,B,mode='constant',value=0);x=A.conv(x);return x
class Upsample(nn.Module):
	def __init__(B,in_channels):A=in_channels;super().__init__();B.conv=nn.Conv2d(A,A,kernel_size=3,stride=1,padding=1)
	def forward(A,x):x=F.interpolate(x,scale_factor=2.,mode=_F);x=A.conv(x);return x
class ResBlock(nn.Module):
	def __init__(A,in_channels,out_channels=_A):
		C=in_channels;B=out_channels;super(ResBlock,A).__init__();A.in_channels=C;A.out_channels=C if B is _A else B;A.norm1=normalize(C);A.conv1=nn.Conv2d(C,B,kernel_size=3,stride=1,padding=1);A.norm2=normalize(B);A.conv2=nn.Conv2d(B,B,kernel_size=3,stride=1,padding=1)
		if A.in_channels!=A.out_channels:A.conv_out=nn.Conv2d(C,B,kernel_size=1,stride=1,padding=0)
	def forward(B,x_in):
		C=x_in;A=C;A=B.norm1(A);A=swish(A);A=B.conv1(A);A=B.norm2(A);A=swish(A);A=B.conv2(A)
		if B.in_channels!=B.out_channels:C=B.conv_out(C)
		return A+C
class AttnBlock(nn.Module):
	def __init__(B,in_channels):A=in_channels;super().__init__();B.in_channels=A;B.norm=normalize(A);B.q=torch.nn.Conv2d(A,A,kernel_size=1,stride=1,padding=0);B.k=torch.nn.Conv2d(A,A,kernel_size=1,stride=1,padding=0);B.v=torch.nn.Conv2d(A,A,kernel_size=1,stride=1,padding=0);B.proj_out=torch.nn.Conv2d(A,A,kernel_size=1,stride=1,padding=0)
	def forward(D,x):A=x;A=D.norm(A);C=D.q(A);J=D.k(A);K=D.v(A);G,E,H,I=C.shape;C=C.reshape(G,E,H*I);C=C.permute(0,2,1);J=J.reshape(G,E,H*I);B=torch.bmm(C,J);B=B*int(E)**-.5;B=F.softmax(B,dim=2);K=K.reshape(G,E,H*I);B=B.permute(0,2,1);A=torch.bmm(K,B);A=A.reshape(G,E,H,I);A=D.proj_out(A);return x+A
class Encoder(nn.Module):
	def __init__(B,in_channels,nf,emb_dim,ch_mult,num_res_blocks,resolution,attn_resolutions):
		G=attn_resolutions;D=ch_mult;super().__init__();B.nf=nf;B.num_resolutions=len(D);B.num_res_blocks=num_res_blocks;B.resolution=resolution;B.attn_resolutions=G;E=B.resolution;I=(1,)+tuple(D);C=[];C.append(nn.Conv2d(in_channels,nf,kernel_size=3,stride=1,padding=1))
		for F in range(B.num_resolutions):
			A=nf*I[F];H=nf*D[F]
			for J in range(B.num_res_blocks):
				C.append(ResBlock(A,H));A=H
				if E in G:C.append(AttnBlock(A))
			if F!=B.num_resolutions-1:C.append(Downsample(A));E=E//2
		C.append(ResBlock(A,A));C.append(AttnBlock(A));C.append(ResBlock(A,A));C.append(normalize(A));C.append(nn.Conv2d(A,emb_dim,kernel_size=3,stride=1,padding=1));B.blocks=nn.ModuleList(C)
	def forward(A,x):
		for B in A.blocks:x=B(x)
		return x
class Generator(nn.Module):
	def __init__(A,nf,emb_dim,ch_mult,res_blocks,img_size,attn_resolutions):
		super().__init__();A.nf=nf;A.ch_mult=ch_mult;A.num_resolutions=len(A.ch_mult);A.num_res_blocks=res_blocks;A.resolution=img_size;A.attn_resolutions=attn_resolutions;A.in_channels=emb_dim;A.out_channels=3;B=A.nf*A.ch_mult[-1];D=A.resolution//2**(A.num_resolutions-1);C=[];C.append(nn.Conv2d(A.in_channels,B,kernel_size=3,stride=1,padding=1));C.append(ResBlock(B,B));C.append(AttnBlock(B));C.append(ResBlock(B,B))
		for E in reversed(range(A.num_resolutions)):
			F=A.nf*A.ch_mult[E]
			for G in range(A.num_res_blocks):
				C.append(ResBlock(B,F));B=F
				if D in A.attn_resolutions:C.append(AttnBlock(B))
			if E!=0:C.append(Upsample(B));D=D*2
		C.append(normalize(B));C.append(nn.Conv2d(B,A.out_channels,kernel_size=3,stride=1,padding=1));A.blocks=nn.ModuleList(C)
	def forward(A,x):
		for B in A.blocks:x=B(x)
		return x
@ARCH_REGISTRY.register()
class VQAutoEncoder(nn.Module):
	def __init__(A,img_size,nf,ch_mult,quantizer=_F,res_blocks=2,attn_resolutions=_A,codebook_size=1024,emb_dim=256,beta=.25,gumbel_straight_through=_D,gumbel_kl_weight=1e-08,model_path=_A):
		F='params_ema';C=emb_dim;B=model_path;super().__init__();D=get_root_logger();A.in_channels=3;A.nf=nf;A.n_blocks=res_blocks;A.codebook_size=codebook_size;A.embed_dim=C;A.ch_mult=ch_mult;A.resolution=img_size;A.attn_resolutions=attn_resolutions or[16];A.quantizer_type=quantizer;A.encoder=Encoder(A.in_channels,A.nf,A.embed_dim,A.ch_mult,A.n_blocks,A.resolution,A.attn_resolutions)
		if A.quantizer_type==_F:A.beta=beta;A.quantize=VectorQuantizer(A.codebook_size,A.embed_dim,A.beta)
		elif A.quantizer_type=='gumbel':A.gumbel_num_hiddens=C;A.straight_through=gumbel_straight_through;A.kl_weight=gumbel_kl_weight;A.quantize=GumbelQuantizer(A.codebook_size,A.embed_dim,A.gumbel_num_hiddens,A.straight_through,A.kl_weight)
		A.generator=Generator(A.nf,A.embed_dim,A.ch_mult,A.n_blocks,A.resolution,A.attn_resolutions)
		if B is not _A:
			E=torch.load(B,map_location=_C)
			if F in E:A.load_state_dict(torch.load(B,map_location=_C)[F]);D.info(f"vqgan is loaded from: {B} [params_ema]")
			elif _E in E:A.load_state_dict(torch.load(B,map_location=_C)[_E]);D.info(f"vqgan is loaded from: {B} [params]")
			else:raise ValueError(_H)
	def forward(A,x):x=A.encoder(x);B,C,D=A.quantize(x);x=A.generator(B);return x,C,D
@ARCH_REGISTRY.register()
class VQGANDiscriminator(nn.Module):
	def __init__(F,nc=3,ndf=64,n_layers=4,model_path=_A):
		I='params_d';G=n_layers;C=model_path;B=ndf;super().__init__();D=[nn.Conv2d(nc,B,kernel_size=4,stride=2,padding=1),nn.LeakyReLU(.2,_B)];A=1;E=1
		for J in range(1,G):E=A;A=min(2**J,8);D+=[nn.Conv2d(B*E,B*A,kernel_size=4,stride=2,padding=1,bias=_D),nn.BatchNorm2d(B*A),nn.LeakyReLU(.2,_B)]
		E=A;A=min(2**G,8);D+=[nn.Conv2d(B*E,B*A,kernel_size=4,stride=1,padding=1,bias=_D),nn.BatchNorm2d(B*A),nn.LeakyReLU(.2,_B)];D+=[nn.Conv2d(B*A,1,kernel_size=4,stride=1,padding=1)];F.main=nn.Sequential(*D)
		if C is not _A:
			H=torch.load(C,map_location=_C)
			if I in H:F.load_state_dict(torch.load(C,map_location=_C)[I])
			elif _E in H:F.load_state_dict(torch.load(C,map_location=_C)[_E])
			else:raise ValueError(_H)
	def forward(A,x):return A.main(x)