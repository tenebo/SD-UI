_F='256'
_E='128'
_D='64'
_C='32'
_B=False
_A=None
import math,torch
from torch import nn,Tensor
import torch.nn.functional as F
from typing import Optional
from modules.codeformer.vqgan_arch import VQAutoEncoder,ResBlock
from basicsr.utils.registry import ARCH_REGISTRY
def calc_mean_std(feat,eps=1e-05):'Calculate mean and std for adaptive_instance_normalization.\n\n    Args:\n        feat (Tensor): 4D tensor.\n        eps (float): A small value added to the variance to avoid\n            divide-by-zero. Default: 1e-5.\n    ';C=feat;D=C.size();assert len(D)==4,'The input feature should be 4D tensor.';A,B=D[:2];E=C.view(A,B,-1).var(dim=2)+eps;F=E.sqrt().view(A,B,1,1);G=C.view(A,B,-1).mean(dim=2).view(A,B,1,1);return G,F
def adaptive_instance_normalization(content_feat,style_feat):'Adaptive instance normalization.\n\n    Adjust the reference features to have the similar color and illuminations\n    as those in the degradate features.\n\n    Args:\n        content_feat (Tensor): The reference feature.\n        style_feat (Tensor): The degradate features.\n    ';B=content_feat;A=B.size();C,D=calc_mean_std(style_feat);E,F=calc_mean_std(B);G=(B-E.expand(A))/F.expand(A);return G*D.expand(A)+C.expand(A)
class PositionEmbeddingSine(nn.Module):
	'\n    This is a more standard version of the position embedding, very similar to the one\n    used by the Attention is all you need paper, generalized to work on images.\n    '
	def __init__(A,num_pos_feats=64,temperature=10000,normalize=_B,scale=_A):
		C=normalize;B=scale;super().__init__();A.num_pos_feats=num_pos_feats;A.temperature=temperature;A.normalize=C
		if B is not _A and C is _B:raise ValueError('normalize should be True if scale is passed')
		if B is _A:B=2*math.pi
		A.scale=B
	def forward(A,x,mask=_A):
		G=mask
		if G is _A:G=torch.zeros((x.size(0),x.size(2),x.size(3)),device=x.device,dtype=torch.bool)
		H=~G;B=H.cumsum(1,dtype=torch.float32);C=H.cumsum(2,dtype=torch.float32)
		if A.normalize:I=1e-06;B=B/(B[:,-1:,:]+I)*A.scale;C=C/(C[:,:,-1:]+I)*A.scale
		D=torch.arange(A.num_pos_feats,dtype=torch.float32,device=x.device);D=A.temperature**(2*(D//2)/A.num_pos_feats);E=C[:,:,:,_A]/D;F=B[:,:,:,_A]/D;E=torch.stack((E[:,:,:,0::2].sin(),E[:,:,:,1::2].cos()),dim=4).flatten(3);F=torch.stack((F[:,:,:,0::2].sin(),F[:,:,:,1::2].cos()),dim=4).flatten(3);J=torch.cat((F,E),dim=3).permute(0,3,1,2);return J
def _get_activation_fn(activation):
	'Return an activation function given a string';A=activation
	if A=='relu':return F.relu
	if A=='gelu':return F.gelu
	if A=='glu':return F.glu
	raise RuntimeError(f"activation should be relu/gelu, not {A}.")
class TransformerSALayer(nn.Module):
	def __init__(A,embed_dim,nhead=8,dim_mlp=2048,dropout=.0,activation='gelu'):D=dim_mlp;C=dropout;B=embed_dim;super().__init__();A.self_attn=nn.MultiheadAttention(B,nhead,dropout=C);A.linear1=nn.Linear(B,D);A.dropout=nn.Dropout(C);A.linear2=nn.Linear(D,B);A.norm1=nn.LayerNorm(B);A.norm2=nn.LayerNorm(B);A.dropout1=nn.Dropout(C);A.dropout2=nn.Dropout(C);A.activation=_get_activation_fn(activation)
	def with_pos_embed(B,tensor,pos):A=tensor;return A if pos is _A else A+pos
	def forward(A,tgt,tgt_mask=_A,tgt_key_padding_mask=_A,query_pos=_A):C=tgt;B=A.norm1(C);D=E=A.with_pos_embed(B,query_pos);B=A.self_attn(D,E,value=B,attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0];C=C+A.dropout1(B);B=A.norm2(C);B=A.linear2(A.dropout(A.activation(A.linear1(B))));C=C+A.dropout2(B);return C
class Fuse_sft_block(nn.Module):
	def __init__(B,in_ch,out_ch):C=in_ch;A=out_ch;super().__init__();B.encode_enc=ResBlock(2*C,A);B.scale=nn.Sequential(nn.Conv2d(C,A,kernel_size=3,padding=1),nn.LeakyReLU(.2,True),nn.Conv2d(A,A,kernel_size=3,padding=1));B.shift=nn.Sequential(nn.Conv2d(C,A,kernel_size=3,padding=1),nn.LeakyReLU(.2,True),nn.Conv2d(A,A,kernel_size=3,padding=1))
	def forward(B,enc_feat,dec_feat,w=1):C=dec_feat;A=enc_feat;A=B.encode_enc(torch.cat([A,C],dim=1));D=B.scale(A);E=B.shift(A);F=w*(C*D+E);G=C+F;return G
@ARCH_REGISTRY.register()
class CodeFormer(VQAutoEncoder):
	def __init__(A,dim_embd=512,n_head=8,n_layers=9,codebook_size=1024,latent_size=256,connect_list=(_C,_D,_E,_F),fix_modules=('quantize','generator')):
		F=fix_modules;E=codebook_size;D='512';C='16';B=dim_embd;super(CodeFormer,A).__init__(512,64,[1,2,2,4,4,8],'nearest',2,[16],E)
		if F is not _A:
			for I in F:
				for J in getattr(A,I).parameters():J.requires_grad=_B
		A.connect_list=connect_list;A.n_layers=n_layers;A.dim_embd=B;A.dim_mlp=B*2;A.position_emb=nn.Parameter(torch.zeros(latent_size,A.dim_embd));A.feat_emb=nn.Linear(256,A.dim_embd);A.ft_layers=nn.Sequential(*[TransformerSALayer(embed_dim=B,nhead=n_head,dim_mlp=A.dim_mlp,dropout=.0)for C in range(A.n_layers)]);A.idx_pred_layer=nn.Sequential(nn.LayerNorm(B),nn.Linear(B,E,bias=_B));A.channels={C:512,_C:256,_D:256,_E:128,_F:128,D:64};A.fuse_encoder_block={D:2,_F:5,_E:8,_D:11,_C:14,C:18};A.fuse_generator_block={C:6,_C:9,_D:12,_E:15,_F:18,D:21};A.fuse_convs_dict=nn.ModuleDict()
		for G in A.connect_list:H=A.channels[G];A.fuse_convs_dict[G]=Fuse_sft_block(H,H)
	def _init_weights(B,module):
		A=module
		if isinstance(A,(nn.Linear,nn.Embedding)):
			A.weight.data.normal_(mean=.0,std=.02)
			if isinstance(A,nn.Linear)and A.bias is not _A:A.bias.data.zero_()
		elif isinstance(A,nn.LayerNorm):A.bias.data.zero_();A.weight.data.fill_(1.)
	def forward(A,x,w=0,detach_16=True,code_only=_B,adain=_B):
		I={};K=[A.fuse_encoder_block[B]for B in A.connect_list]
		for(E,G)in enumerate(A.encoder.blocks):
			x=G(x)
			if E in K:I[str(x.shape[-1])]=x.clone()
		D=x;L=A.position_emb.unsqueeze(1).repeat(1,x.shape[0],1);M=A.feat_emb(D.flatten(2).permute(2,0,1));H=M
		for N in A.ft_layers:H=N(H,query_pos=L)
		B=A.idx_pred_layer(H);B=B.permute(1,0,2)
		if code_only:return B,D
		O=F.softmax(B,dim=2);S,P=torch.topk(O,1,dim=2);C=A.quantize.get_codebook_feat(P,shape=[x.shape[0],16,16,256])
		if detach_16:C=C.detach()
		if adain:C=adaptive_instance_normalization(C,D)
		x=C;Q=[A.fuse_generator_block[B]for B in A.connect_list]
		for(E,G)in enumerate(A.generator.blocks):
			x=G(x)
			if E in Q:
				J=str(x.shape[-1])
				if w>0:x=A.fuse_convs_dict[J](I[J].detach(),x,w)
		R=x;return R,B,D