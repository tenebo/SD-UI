_H='nearest+conv'
_G='pixelshuffle'
_F='pixelshuffledirect'
_E='1conv'
_D=False
_C=True
_B=.0
_A=None
import math,torch,torch.nn as nn,torch.nn.functional as F,torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath,to_2tuple,trunc_normal_
class Mlp(nn.Module):
	def __init__(A,in_features,hidden_features=_A,out_features=_A,act_layer=nn.GELU,drop=_B):C=out_features;D=in_features;B=hidden_features;super().__init__();C=C or D;B=B or D;A.fc1=nn.Linear(D,B);A.act=act_layer();A.fc2=nn.Linear(B,C);A.drop=nn.Dropout(drop)
	def forward(A,x):x=A.fc1(x);x=A.act(x);x=A.drop(x);x=A.fc2(x);x=A.drop(x);return x
def window_partition(x,window_size):'\n    Args:\n        x: (B, H, W, C)\n        window_size (int): window size\n\n    Returns:\n        windows: (num_windows*B, window_size, window_size, C)\n    ';A=window_size;C,D,E,B=x.shape;x=x.view(C,D//A,A,E//A,A,B);F=x.permute(0,1,3,2,4,5).contiguous().view(-1,A,A,B);return F
def window_reverse(windows,window_size,H,W):'\n    Args:\n        windows: (num_windows*B, window_size, window_size, C)\n        window_size (int): Window size\n        H (int): Height of image\n        W (int): Width of image\n\n    Returns:\n        x: (B, H, W, C)\n    ';C=windows;A=window_size;D=int(C.shape[0]/(H*W/A/A));B=C.view(D,H//A,W//A,A,A,-1);B=B.permute(0,1,3,2,4,5).contiguous().view(D,H,W,-1);return B
class WindowAttention(nn.Module):
	' Window based multi-head self attention (W-MSA) module with relative position bias.\n    It supports both of shifted and non-shifted window.\n\n    Args:\n        dim (int): Number of input channels.\n        window_size (tuple[int]): The height and width of the window.\n        num_heads (int): Number of attention heads.\n        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True\n        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set\n        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0\n        proj_drop (float, optional): Dropout ratio of output. Default: 0.0\n    '
	def __init__(A,dim,window_size,num_heads,qkv_bias=_C,qk_scale=_A,attn_drop=_B,proj_drop=_B):D=num_heads;E=window_size;B=dim;super().__init__();A.dim=B;A.window_size=E;A.num_heads=D;G=B//D;A.scale=qk_scale or G**-.5;A.relative_position_bias_table=nn.Parameter(torch.zeros((2*E[0]-1)*(2*E[1]-1),D));H=torch.arange(A.window_size[0]);I=torch.arange(A.window_size[1]);J=torch.stack(torch.meshgrid([H,I]));F=torch.flatten(J,1);C=F[:,:,_A]-F[:,_A,:];C=C.permute(1,2,0).contiguous();C[:,:,0]+=A.window_size[0]-1;C[:,:,1]+=A.window_size[1]-1;C[:,:,0]*=2*A.window_size[1]-1;K=C.sum(-1);A.register_buffer('relative_position_index',K);A.qkv=nn.Linear(B,B*3,bias=qkv_bias);A.attn_drop=nn.Dropout(attn_drop);A.proj=nn.Linear(B,B);A.proj_drop=nn.Dropout(proj_drop);trunc_normal_(A.relative_position_bias_table,std=.02);A.softmax=nn.Softmax(dim=-1)
	def forward(A,x,mask=_A):
		'\n        Args:\n            x: input features with shape of (num_windows*B, N, C)\n            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None\n        ';D=mask;E,C,I=x.shape;F=A.qkv(x).reshape(E,C,3,A.num_heads,I//A.num_heads).permute(2,0,3,1,4);G,K,L=F[0],F[1],F[2];G=G*A.scale;B=G@K.transpose(-2,-1);H=A.relative_position_bias_table[A.relative_position_index.view(-1)].view(A.window_size[0]*A.window_size[1],A.window_size[0]*A.window_size[1],-1);H=H.permute(2,0,1).contiguous();B=B+H.unsqueeze(0)
		if D is not _A:J=D.shape[0];B=B.view(E//J,J,A.num_heads,C,C)+D.unsqueeze(1).unsqueeze(0);B=B.view(-1,A.num_heads,C,C);B=A.softmax(B)
		else:B=A.softmax(B)
		B=A.attn_drop(B);x=(B@L).transpose(1,2).reshape(E,C,I);x=A.proj(x);x=A.proj_drop(x);return x
	def extra_repr(A):return f"dim={A.dim}, window_size={A.window_size}, num_heads={A.num_heads}"
	def flops(A,N):B=0;B+=N*A.dim*3*A.dim;B+=A.num_heads*N*(A.dim//A.num_heads)*N;B+=A.num_heads*N*N*(A.dim//A.num_heads);B+=N*A.dim*A.dim;return B
class SwinTransformerBlock(nn.Module):
	' Swin Transformer Block.\n\n    Args:\n        dim (int): Number of input channels.\n        input_resolution (tuple[int]): Input resolution.\n        num_heads (int): Number of attention heads.\n        window_size (int): Window size.\n        shift_size (int): Shift size for SW-MSA.\n        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.\n        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True\n        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.\n        drop (float, optional): Dropout rate. Default: 0.0\n        attn_drop (float, optional): Attention dropout rate. Default: 0.0\n        drop_path (float, optional): Stochastic depth rate. Default: 0.0\n        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU\n        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm\n    '
	def __init__(A,dim,input_resolution,num_heads,window_size=7,shift_size=0,mlp_ratio=4.,qkv_bias=_C,qk_scale=_A,drop=_B,attn_drop=_B,drop_path=_B,act_layer=nn.GELU,norm_layer=nn.LayerNorm):
		C=norm_layer;D=drop_path;E=mlp_ratio;F=num_heads;B=dim;super().__init__();A.dim=B;A.input_resolution=input_resolution;A.num_heads=F;A.window_size=window_size;A.shift_size=shift_size;A.mlp_ratio=E
		if min(A.input_resolution)<=A.window_size:A.shift_size=0;A.window_size=min(A.input_resolution)
		assert 0<=A.shift_size<A.window_size,'shift_size must in 0-window_size';A.norm1=C(B);A.attn=WindowAttention(B,window_size=to_2tuple(A.window_size),num_heads=F,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop);A.drop_path=DropPath(D)if D>_B else nn.Identity();A.norm2=C(B);H=int(B*E);A.mlp=Mlp(in_features=B,hidden_features=H,act_layer=act_layer,drop=drop)
		if A.shift_size>0:G=A.calculate_mask(A.input_resolution)
		else:G=_A
		A.register_buffer('attn_mask',G)
	def calculate_mask(A,x_size):
		F,G=x_size;D=torch.zeros((1,F,G,1));H=slice(0,-A.window_size),slice(-A.window_size,-A.shift_size),slice(-A.shift_size,_A);I=slice(0,-A.window_size),slice(-A.window_size,-A.shift_size),slice(-A.shift_size,_A);E=0
		for J in H:
			for K in I:D[:,J,K,:]=E;E+=1
		C=window_partition(D,A.window_size);C=C.view(-1,A.window_size*A.window_size);B=C.unsqueeze(1)-C.unsqueeze(2);B=B.masked_fill(B!=0,float(-1e2)).masked_fill(B==0,float(_B));return B
	def forward(A,x,x_size):
		F=x_size;G,H=F;I,K,C=x.shape;J=x;x=A.norm1(x);x=x.view(I,G,H,C)
		if A.shift_size>0:B=torch.roll(x,shifts=(-A.shift_size,-A.shift_size),dims=(1,2))
		else:B=x
		D=window_partition(B,A.window_size);D=D.view(-1,A.window_size*A.window_size,C)
		if A.input_resolution==F:E=A.attn(D,mask=A.attn_mask)
		else:E=A.attn(D,mask=A.calculate_mask(F).to(x.device))
		E=E.view(-1,A.window_size,A.window_size,C);B=window_reverse(E,A.window_size,G,H)
		if A.shift_size>0:x=torch.roll(B,shifts=(A.shift_size,A.shift_size),dims=(1,2))
		else:x=B
		x=x.view(I,G*H,C);x=J+A.drop_path(x);x=x+A.drop_path(A.mlp(A.norm2(x)));return x
	def extra_repr(A):return f"dim={A.dim}, input_resolution={A.input_resolution}, num_heads={A.num_heads}, window_size={A.window_size}, shift_size={A.shift_size}, mlp_ratio={A.mlp_ratio}"
	def flops(A):B=0;C,D=A.input_resolution;B+=A.dim*C*D;E=C*D/A.window_size/A.window_size;B+=E*A.attn.flops(A.window_size*A.window_size);B+=2*C*D*A.dim*A.dim*A.mlp_ratio;B+=A.dim*C*D;return B
class PatchMerging(nn.Module):
	' Patch Merging Layer.\n\n    Args:\n        input_resolution (tuple[int]): Resolution of input feature.\n        dim (int): Number of input channels.\n        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm\n    '
	def __init__(A,input_resolution,dim,norm_layer=nn.LayerNorm):B=dim;super().__init__();A.input_resolution=input_resolution;A.dim=B;A.reduction=nn.Linear(4*B,2*B,bias=_D);A.norm=norm_layer(4*B)
	def forward(C,x):'\n        x: B, H*W, C\n        ';A,B=C.input_resolution;D,F,E=x.shape;assert F==A*B,'input feature has wrong size';assert A%2==0 and B%2==0,f"x size ({A}*{B}) are not even.";x=x.view(D,A,B,E);G=x[:,0::2,0::2,:];H=x[:,1::2,0::2,:];I=x[:,0::2,1::2,:];J=x[:,1::2,1::2,:];x=torch.cat([G,H,I,J],-1);x=x.view(D,-1,4*E);x=C.norm(x);x=C.reduction(x);return x
	def extra_repr(A):return f"input_resolution={A.input_resolution}, dim={A.dim}"
	def flops(A):B,C=A.input_resolution;D=B*C*A.dim;D+=B//2*(C//2)*4*A.dim*2*A.dim;return D
class BasicLayer(nn.Module):
	' A basic Swin Transformer layer for one stage.\n\n    Args:\n        dim (int): Number of input channels.\n        input_resolution (tuple[int]): Input resolution.\n        depth (int): Number of blocks.\n        num_heads (int): Number of attention heads.\n        window_size (int): Local window size.\n        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.\n        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True\n        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.\n        drop (float, optional): Dropout rate. Default: 0.0\n        attn_drop (float, optional): Attention dropout rate. Default: 0.0\n        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0\n        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm\n        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None\n        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.\n    '
	def __init__(A,dim,input_resolution,depth,num_heads,window_size,mlp_ratio=4.,qkv_bias=_C,qk_scale=_A,drop=_B,attn_drop=_B,drop_path=_B,norm_layer=nn.LayerNorm,downsample=_A,use_checkpoint=_D):
		E=downsample;F=norm_layer;G=window_size;H=depth;B=drop_path;C=input_resolution;D=dim;super().__init__();A.dim=D;A.input_resolution=C;A.depth=H;A.use_checkpoint=use_checkpoint;A.blocks=nn.ModuleList([SwinTransformerBlock(dim=D,input_resolution=C,num_heads=num_heads,window_size=G,shift_size=0 if A%2==0 else G//2,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,drop_path=B[A]if isinstance(B,list)else B,norm_layer=F)for A in range(H)])
		if E is not _A:A.downsample=E(C,dim=D,norm_layer=F)
		else:A.downsample=_A
	def forward(A,x,x_size):
		B=x_size
		for C in A.blocks:
			if A.use_checkpoint:x=checkpoint.checkpoint(C,x,B)
			else:x=C(x,B)
		if A.downsample is not _A:x=A.downsample(x)
		return x
	def extra_repr(A):return f"dim={A.dim}, input_resolution={A.input_resolution}, depth={A.depth}"
	def flops(A):
		B=0
		for C in A.blocks:B+=C.flops()
		if A.downsample is not _A:B+=A.downsample.flops()
		return B
class RSTB(nn.Module):
	'Residual Swin Transformer Block (RSTB).\n\n    Args:\n        dim (int): Number of input channels.\n        input_resolution (tuple[int]): Input resolution.\n        depth (int): Number of blocks.\n        num_heads (int): Number of attention heads.\n        window_size (int): Local window size.\n        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.\n        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True\n        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.\n        drop (float, optional): Dropout rate. Default: 0.0\n        attn_drop (float, optional): Attention dropout rate. Default: 0.0\n        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0\n        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm\n        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None\n        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.\n        img_size: Input image size.\n        patch_size: Patch size.\n        resi_connection: The convolutional block before residual connection.\n    '
	def __init__(B,dim,input_resolution,depth,num_heads,window_size,mlp_ratio=4.,qkv_bias=_C,qk_scale=_A,drop=_B,attn_drop=_B,drop_path=_B,norm_layer=nn.LayerNorm,downsample=_A,use_checkpoint=_D,img_size=224,patch_size=4,resi_connection=_E):
		C=resi_connection;D=patch_size;E=img_size;F=input_resolution;A=dim;super(RSTB,B).__init__();B.dim=A;B.input_resolution=F;B.residual_group=BasicLayer(dim=A,input_resolution=F,depth=depth,num_heads=num_heads,window_size=window_size,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop=drop,attn_drop=attn_drop,drop_path=drop_path,norm_layer=norm_layer,downsample=downsample,use_checkpoint=use_checkpoint)
		if C==_E:B.conv=nn.Conv2d(A,A,3,1,1)
		elif C=='3conv':B.conv=nn.Sequential(nn.Conv2d(A,A//4,3,1,1),nn.LeakyReLU(negative_slope=.2,inplace=_C),nn.Conv2d(A//4,A//4,1,1,0),nn.LeakyReLU(negative_slope=.2,inplace=_C),nn.Conv2d(A//4,A,3,1,1))
		B.patch_embed=PatchEmbed(img_size=E,patch_size=D,in_chans=0,embed_dim=A,norm_layer=_A);B.patch_unembed=PatchUnEmbed(img_size=E,patch_size=D,in_chans=0,embed_dim=A,norm_layer=_A)
	def forward(A,x,x_size):B=x_size;return A.patch_embed(A.conv(A.patch_unembed(A.residual_group(x,B),B)))+x
	def flops(A):B=0;B+=A.residual_group.flops();C,D=A.input_resolution;B+=C*D*A.dim*A.dim*9;B+=A.patch_embed.flops();B+=A.patch_unembed.flops();return B
class PatchEmbed(nn.Module):
	' Image to Patch Embedding\n\n    Args:\n        img_size (int): Image size.  Default: 224.\n        patch_size (int): Patch token size. Default: 4.\n        in_chans (int): Number of input image channels. Default: 3.\n        embed_dim (int): Number of linear projection output channels. Default: 96.\n        norm_layer (nn.Module, optional): Normalization layer. Default: None\n    '
	def __init__(A,img_size=224,patch_size=4,in_chans=3,embed_dim=96,norm_layer=_A):
		E=norm_layer;F=embed_dim;B=patch_size;C=img_size;super().__init__();C=to_2tuple(C);B=to_2tuple(B);D=[C[0]//B[0],C[1]//B[1]];A.img_size=C;A.patch_size=B;A.patches_resolution=D;A.num_patches=D[0]*D[1];A.in_chans=in_chans;A.embed_dim=F
		if E is not _A:A.norm=E(F)
		else:A.norm=_A
	def forward(A,x):
		x=x.flatten(2).transpose(1,2)
		if A.norm is not _A:x=A.norm(x)
		return x
	def flops(A):
		B=0;C,D=A.img_size
		if A.norm is not _A:B+=C*D*A.embed_dim
		return B
class PatchUnEmbed(nn.Module):
	' Image to Patch Unembedding\n\n    Args:\n        img_size (int): Image size.  Default: 224.\n        patch_size (int): Patch token size. Default: 4.\n        in_chans (int): Number of input image channels. Default: 3.\n        embed_dim (int): Number of linear projection output channels. Default: 96.\n        norm_layer (nn.Module, optional): Normalization layer. Default: None\n    '
	def __init__(A,img_size=224,patch_size=4,in_chans=3,embed_dim=96,norm_layer=_A):B=patch_size;C=img_size;super().__init__();C=to_2tuple(C);B=to_2tuple(B);D=[C[0]//B[0],C[1]//B[1]];A.img_size=C;A.patch_size=B;A.patches_resolution=D;A.num_patches=D[0]*D[1];A.in_chans=in_chans;A.embed_dim=embed_dim
	def forward(B,x,x_size):A=x_size;C,D,E=x.shape;x=x.transpose(1,2).view(C,B.embed_dim,A[0],A[1]);return x
	def flops(B):A=0;return A
class Upsample(nn.Sequential):
	'Upsample module.\n\n    Args:\n        scale (int): Scale factor. Supported scales: 2^n and 3.\n        num_feat (int): Channel number of intermediate features.\n    '
	def __init__(D,scale,num_feat):
		C=num_feat;A=scale;B=[]
		if A&A-1==0:
			for E in range(int(math.log(A,2))):B.append(nn.Conv2d(C,4*C,3,1,1));B.append(nn.PixelShuffle(2))
		elif A==3:B.append(nn.Conv2d(C,9*C,3,1,1));B.append(nn.PixelShuffle(3))
		else:raise ValueError(f"scale {A} is not supported. Supported scales: 2^n and 3.")
		super(Upsample,D).__init__(*B)
class UpsampleOneStep(nn.Sequential):
	'UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)\n       Used in lightweight SR to save parameters.\n\n    Args:\n        scale (int): Scale factor. Supported scales: 2^n and 3.\n        num_feat (int): Channel number of intermediate features.\n\n    '
	def __init__(A,scale,num_feat,num_out_ch,input_resolution=_A):C=num_feat;D=scale;A.num_feat=C;A.input_resolution=input_resolution;B=[];B.append(nn.Conv2d(C,D**2*num_out_ch,3,1,1));B.append(nn.PixelShuffle(D));super(UpsampleOneStep,A).__init__(*B)
	def flops(A):B,C=A.input_resolution;D=B*C*A.num_feat*3*9;return D
class SwinIR(nn.Module):
	" SwinIR\n        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.\n\n    Args:\n        img_size (int | tuple(int)): Input image size. Default 64\n        patch_size (int | tuple(int)): Patch size. Default: 1\n        in_chans (int): Number of input image channels. Default: 3\n        embed_dim (int): Patch embedding dimension. Default: 96\n        depths (tuple(int)): Depth of each Swin Transformer layer.\n        num_heads (tuple(int)): Number of attention heads in different layers.\n        window_size (int): Window size. Default: 7\n        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4\n        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True\n        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None\n        drop_rate (float): Dropout rate. Default: 0\n        attn_drop_rate (float): Attention dropout rate. Default: 0\n        drop_path_rate (float): Stochastic depth rate. Default: 0.1\n        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.\n        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False\n        patch_norm (bool): If True, add normalization after patch embedding. Default: True\n        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False\n        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction\n        img_range: Image range. 1. or 255.\n        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None\n        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'\n    "
	def __init__(A,img_size=64,patch_size=1,in_chans=3,embed_dim=96,depths=(6,6,6,6),num_heads=(6,6,6,6),window_size=7,mlp_ratio=4.,qkv_bias=_C,qk_scale=_A,drop_rate=_B,attn_drop_rate=_B,drop_path_rate=.1,norm_layer=nn.LayerNorm,ape=_D,patch_norm=_C,use_checkpoint=_D,upscale=2,img_range=1.,upsampler='',resi_connection=_E,**U):
		N=drop_rate;O=window_size;I=resi_connection;J=upscale;K=in_chans;L=patch_size;M=img_size;F=norm_layer;D=depths;B=embed_dim;super(SwinIR,A).__init__();P=K;G=K;C=64;A.img_range=img_range
		if K==3:Q=.4488,.4371,.404;A.mean=torch.Tensor(Q).view(1,3,1,1)
		else:A.mean=torch.zeros(1,1,1,1)
		A.upscale=J;A.upsampler=upsampler;A.window_size=O;A.conv_first=nn.Conv2d(P,B,3,1,1);A.num_layers=len(D);A.embed_dim=B;A.ape=ape;A.patch_norm=patch_norm;A.num_features=B;A.mlp_ratio=mlp_ratio;A.patch_embed=PatchEmbed(img_size=M,patch_size=L,in_chans=B,embed_dim=B,norm_layer=F if A.patch_norm else _A);R=A.patch_embed.num_patches;E=A.patch_embed.patches_resolution;A.patches_resolution=E;A.patch_unembed=PatchUnEmbed(img_size=M,patch_size=L,in_chans=B,embed_dim=B,norm_layer=F if A.patch_norm else _A)
		if A.ape:A.absolute_pos_embed=nn.Parameter(torch.zeros(1,R,B));trunc_normal_(A.absolute_pos_embed,std=.02)
		A.pos_drop=nn.Dropout(p=N);S=[A.item()for A in torch.linspace(0,drop_path_rate,sum(D))];A.layers=nn.ModuleList()
		for H in range(A.num_layers):T=RSTB(dim=B,input_resolution=(E[0],E[1]),depth=D[H],num_heads=num_heads[H],window_size=O,mlp_ratio=A.mlp_ratio,qkv_bias=qkv_bias,qk_scale=qk_scale,drop=N,attn_drop=attn_drop_rate,drop_path=S[sum(D[:H]):sum(D[:H+1])],norm_layer=F,downsample=_A,use_checkpoint=use_checkpoint,img_size=M,patch_size=L,resi_connection=I);A.layers.append(T)
		A.norm=F(A.num_features)
		if I==_E:A.conv_after_body=nn.Conv2d(B,B,3,1,1)
		elif I=='3conv':A.conv_after_body=nn.Sequential(nn.Conv2d(B,B//4,3,1,1),nn.LeakyReLU(negative_slope=.2,inplace=_C),nn.Conv2d(B//4,B//4,1,1,0),nn.LeakyReLU(negative_slope=.2,inplace=_C),nn.Conv2d(B//4,B,3,1,1))
		if A.upsampler==_G:A.conv_before_upsample=nn.Sequential(nn.Conv2d(B,C,3,1,1),nn.LeakyReLU(inplace=_C));A.upsample=Upsample(J,C);A.conv_last=nn.Conv2d(C,G,3,1,1)
		elif A.upsampler==_F:A.upsample=UpsampleOneStep(J,B,G,(E[0],E[1]))
		elif A.upsampler==_H:
			A.conv_before_upsample=nn.Sequential(nn.Conv2d(B,C,3,1,1),nn.LeakyReLU(inplace=_C));A.conv_up1=nn.Conv2d(C,C,3,1,1)
			if A.upscale==4:A.conv_up2=nn.Conv2d(C,C,3,1,1)
			A.conv_hr=nn.Conv2d(C,C,3,1,1);A.conv_last=nn.Conv2d(C,G,3,1,1);A.lrelu=nn.LeakyReLU(negative_slope=.2,inplace=_C)
		else:A.conv_last=nn.Conv2d(B,G,3,1,1)
		A.apply(A._init_weights)
	def _init_weights(A,m):
		if isinstance(m,nn.Linear):
			trunc_normal_(m.weight,std=.02)
			if isinstance(m,nn.Linear)and m.bias is not _A:nn.init.constant_(m.bias,0)
		elif isinstance(m,nn.LayerNorm):nn.init.constant_(m.bias,0);nn.init.constant_(m.weight,1.)
	@torch.jit.ignore
	def no_weight_decay(self):return{'absolute_pos_embed'}
	@torch.jit.ignore
	def no_weight_decay_keywords(self):return{'relative_position_bias_table'}
	def check_image_size(A,x):B,B,C,D=x.size();E=(A.window_size-C%A.window_size)%A.window_size;G=(A.window_size-D%A.window_size)%A.window_size;x=F.pad(x,(0,G,0,E),'reflect');return x
	def forward_features(A,x):
		B=x.shape[2],x.shape[3];x=A.patch_embed(x)
		if A.ape:x=x+A.absolute_pos_embed
		x=A.pos_drop(x)
		for C in A.layers:x=C(x,B)
		x=A.norm(x);x=A.patch_unembed(x,B);return x
	def forward(A,x):
		B='nearest';D,E=x.shape[2:];x=A.check_image_size(x);A.mean=A.mean.type_as(x);x=(x-A.mean)*A.img_range
		if A.upsampler==_G:x=A.conv_first(x);x=A.conv_after_body(A.forward_features(x))+x;x=A.conv_before_upsample(x);x=A.conv_last(A.upsample(x))
		elif A.upsampler==_F:x=A.conv_first(x);x=A.conv_after_body(A.forward_features(x))+x;x=A.upsample(x)
		elif A.upsampler==_H:
			x=A.conv_first(x);x=A.conv_after_body(A.forward_features(x))+x;x=A.conv_before_upsample(x);x=A.lrelu(A.conv_up1(torch.nn.functional.interpolate(x,scale_factor=2,mode=B)))
			if A.upscale==4:x=A.lrelu(A.conv_up2(torch.nn.functional.interpolate(x,scale_factor=2,mode=B)))
			x=A.conv_last(A.lrelu(A.conv_hr(x)))
		else:C=A.conv_first(x);F=A.conv_after_body(A.forward_features(C))+C;x=x+A.conv_last(F)
		x=x/A.img_range+A.mean;return x[:,:,:D*A.upscale,:E*A.upscale]
	def flops(A):
		B=0;C,D=A.patches_resolution;B+=C*D*3*A.embed_dim*9;B+=A.patch_embed.flops()
		for E in A.layers:B+=E.flops()
		B+=C*D*3*A.embed_dim*A.embed_dim;B+=A.upsample.flops();return B
if __name__=='__main__':upscale=4;window_size=8;height=(1024//upscale//window_size+1)*window_size;width=(720//upscale//window_size+1)*window_size;model=SwinIR(upscale=2,img_size=(height,width),window_size=window_size,img_range=1.,depths=[6,6,6,6],embed_dim=60,num_heads=[6,6,6,6],mlp_ratio=2,upsampler=_F);print(model);print(height,width,model.flops()/1e9);x=torch.randn((1,3,height,width));x=model(x);print(x.shape)