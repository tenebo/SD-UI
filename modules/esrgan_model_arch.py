_L='Conv3D'
_K='upconv'
_J='nearest'
_I='prelu'
_H='relu'
_G='Conv2D'
_F='leakyrelu'
_E='CNA'
_D=True
_C='zero'
_B=False
_A=None
from collections import OrderedDict
import math,torch,torch.nn as nn,torch.nn.functional as F
class RRDBNet(nn.Module):
	def __init__(C,in_nc,out_nc,nf,nb,nr=3,gc=32,upscale=4,norm_type=_A,act_type=_F,mode=_E,upsample_mode=_K,convtype=_G,finalact=_A,gaussian_noise=_B,plus=_B):
		I=finalact;J=norm_type;F=upsample_mode;G=upscale;D=act_type;E=in_nc;B=convtype;A=nf;super(RRDBNet,C).__init__();K=int(math.log(G,2))
		if G==3:K=1
		C.resrgan_scale=0
		if E%16==0:C.resrgan_scale=1
		elif E!=4 and E%4==0:C.resrgan_scale=2
		M=conv_block(E,A,kernel_size=3,norm_type=_A,act_type=_A,convtype=B);N=[RRDB(A,nr,kernel_size=3,gc=32,stride=1,bias=1,pad_type=_C,norm_type=J,act_type=D,mode=_E,convtype=B,gaussian_noise=gaussian_noise,plus=plus)for C in range(nb)];O=conv_block(A,A,kernel_size=3,norm_type=J,act_type=_A,mode=mode,convtype=B)
		if F==_K:H=upconv_block
		elif F=='pixelshuffle':H=pixelshuffle_block
		else:raise NotImplementedError(f"upsample mode [{F}] is not found")
		if G==3:L=H(A,A,3,act_type=D,convtype=B)
		else:L=[H(A,A,act_type=D,convtype=B)for C in range(K)]
		P=conv_block(A,A,kernel_size=3,norm_type=_A,act_type=D,convtype=B);Q=conv_block(A,out_nc,kernel_size=3,norm_type=_A,act_type=_A,convtype=B);R=act(I)if I else _A;C.model=sequential(M,ShortcutBlock(sequential(*N,O)),*L,P,Q,R)
	def forward(A,x,outm=_A):
		if A.resrgan_scale==1:B=pixel_unshuffle(x,scale=4)
		elif A.resrgan_scale==2:B=pixel_unshuffle(x,scale=2)
		else:B=x
		return A.model(B)
class RRDB(nn.Module):
	'\n    Residual in Residual Dense Block\n    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)\n    '
	def __init__(A,nf,nr=3,kernel_size=3,gc=32,stride=1,bias=1,pad_type=_C,norm_type=_A,act_type=_F,mode=_E,convtype=_G,spectral_norm=_B,gaussian_noise=_B,plus=_B):
		B=plus;C=gaussian_noise;D=spectral_norm;E=convtype;F=mode;G=act_type;H=norm_type;I=pad_type;J=bias;K=stride;L=kernel_size;super(RRDB,A).__init__()
		if nr==3:A.RDB1=ResidualDenseBlock_5C(nf,L,gc,K,J,I,H,G,F,E,spectral_norm=D,gaussian_noise=C,plus=B);A.RDB2=ResidualDenseBlock_5C(nf,L,gc,K,J,I,H,G,F,E,spectral_norm=D,gaussian_noise=C,plus=B);A.RDB3=ResidualDenseBlock_5C(nf,L,gc,K,J,I,H,G,F,E,spectral_norm=D,gaussian_noise=C,plus=B)
		else:M=[ResidualDenseBlock_5C(nf,L,gc,K,J,I,H,G,F,E,spectral_norm=D,gaussian_noise=C,plus=B)for A in range(nr)];A.RDBs=nn.Sequential(*M)
	def forward(B,x):
		if hasattr(B,'RDB1'):A=B.RDB1(x);A=B.RDB2(A);A=B.RDB3(A)
		else:A=B.RDBs(x)
		return A*.2+x
class ResidualDenseBlock_5C(nn.Module):
	'\n    Residual Dense Block\n    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)\n    Modified options that can be used:\n        - "Partial Convolution based Padding" arXiv:1811.11718\n        - "Spectral normalization" arXiv:1802.05957\n        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.\n            {Rakotonirina} and A. {Rasoanaivo}\n    '
	def __init__(B,nf=64,kernel_size=3,gc=32,stride=1,bias=1,pad_type=_C,norm_type=_A,act_type=_F,mode=_E,convtype=_G,spectral_norm=_B,gaussian_noise=_B,plus=_B):
		L=kernel_size;E=spectral_norm;F=convtype;G=act_type;H=norm_type;I=pad_type;J=bias;K=stride;D=mode;C=nf;A=gc;super(ResidualDenseBlock_5C,B).__init__();B.noise=GaussianNoise()if gaussian_noise else _A;B.conv1x1=conv1x1(C,A)if plus else _A;B.conv1=conv_block(C,A,L,K,bias=J,pad_type=I,norm_type=H,act_type=G,mode=D,convtype=F,spectral_norm=E);B.conv2=conv_block(C+A,A,L,K,bias=J,pad_type=I,norm_type=H,act_type=G,mode=D,convtype=F,spectral_norm=E);B.conv3=conv_block(C+2*A,A,L,K,bias=J,pad_type=I,norm_type=H,act_type=G,mode=D,convtype=F,spectral_norm=E);B.conv4=conv_block(C+3*A,A,L,K,bias=J,pad_type=I,norm_type=H,act_type=G,mode=D,convtype=F,spectral_norm=E)
		if D==_E:M=_A
		else:M=G
		B.conv5=conv_block(C+4*A,C,3,K,bias=J,pad_type=I,norm_type=H,act_type=M,mode=D,convtype=F,spectral_norm=E)
	def forward(A,x):
		C=A.conv1(x);B=A.conv2(torch.cat((x,C),1))
		if A.conv1x1:B=B+A.conv1x1(x)
		E=A.conv3(torch.cat((x,C,B),1));D=A.conv4(torch.cat((x,C,B,E),1))
		if A.conv1x1:D=D+B
		F=A.conv5(torch.cat((x,C,B,E,D),1))
		if A.noise:return A.noise(F.mul(.2)+x)
		else:return F*.2+x
class GaussianNoise(nn.Module):
	def __init__(A,sigma=.1,is_relative_detach=_B):super().__init__();A.sigma=sigma;A.is_relative_detach=is_relative_detach;A.noise=torch.tensor(0,dtype=torch.float)
	def forward(A,x):
		if A.training and A.sigma!=0:A.noise=A.noise.to(x.device);B=A.sigma*x.detach()if A.is_relative_detach else A.sigma*x;C=A.noise.repeat(*x.size()).normal_()*B;x=x+C
		return x
def conv1x1(in_planes,out_planes,stride=1):return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=_B)
class SRVGGNetCompact(nn.Module):
	'A compact VGG-style network structure for super-resolution.\n    This class is copied from https://github.com/xinntao/Real-ESRGAN\n    '
	def __init__(A,num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=16,upscale=4,act_type=_I):
		F=num_conv;G=num_out_ch;H=num_in_ch;E=upscale;B=act_type;C=num_feat;super(SRVGGNetCompact,A).__init__();A.num_in_ch=H;A.num_out_ch=G;A.num_feat=C;A.num_conv=F;A.upscale=E;A.act_type=B;A.body=nn.ModuleList();A.body.append(nn.Conv2d(H,C,3,1,1))
		if B==_H:D=nn.ReLU(inplace=_D)
		elif B==_I:D=nn.PReLU(num_parameters=C)
		elif B==_F:D=nn.LeakyReLU(negative_slope=.1,inplace=_D)
		A.body.append(D)
		for I in range(F):
			A.body.append(nn.Conv2d(C,C,3,1,1))
			if B==_H:D=nn.ReLU(inplace=_D)
			elif B==_I:D=nn.PReLU(num_parameters=C)
			elif B==_F:D=nn.LeakyReLU(negative_slope=.1,inplace=_D)
			A.body.append(D)
		A.body.append(nn.Conv2d(C,G*E*E,3,1,1));A.upsampler=nn.PixelShuffle(E)
	def forward(B,x):
		A=x
		for C in range(0,len(B.body)):A=B.body[C](A)
		A=B.upsampler(A);D=F.interpolate(x,scale_factor=B.upscale,mode=_J);A+=D;return A
class Upsample(nn.Module):
	'Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.\n    The input data is assumed to be of the form\n    `minibatch x channels x [optional depth] x [optional height] x width`.\n    '
	def __init__(A,size=_A,scale_factor=_A,mode=_J,align_corners=_A):
		B=scale_factor;super(Upsample,A).__init__()
		if isinstance(B,tuple):A.scale_factor=tuple(float(A)for A in B)
		else:A.scale_factor=float(B)if B else _A
		A.mode=mode;A.size=size;A.align_corners=align_corners
	def forward(A,x):return nn.functional.interpolate(x,size=A.size,scale_factor=A.scale_factor,mode=A.mode,align_corners=A.align_corners)
	def extra_repr(A):
		if A.scale_factor is not _A:B=f"scale_factor={A.scale_factor}"
		else:B=f"size={A.size}"
		B+=f", mode={A.mode}";return B
def pixel_unshuffle(x,scale):' Pixel unshuffle.\n    Args:\n        x (Tensor): Input feature with shape (b, c, hh, hw).\n        scale (int): Downsample ratio.\n    Returns:\n        Tensor: the pixel unshuffled feature.\n    ';A=scale;B,C,D,E=x.size();H=C*A**2;assert D%A==0 and E%A==0;F=D//A;G=E//A;I=x.view(B,C,F,A,G,A);return I.permute(0,1,3,5,2,4).reshape(B,H,F,G)
def pixelshuffle_block(in_nc,out_nc,upscale_factor=2,kernel_size=3,stride=1,bias=_D,pad_type=_C,norm_type=_A,act_type=_H,convtype=_G):'\n    Pixel shuffle layer\n    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional\n    Neural Network, CVPR17)\n    ';A=act_type;B=norm_type;C=upscale_factor;D=out_nc;E=conv_block(in_nc,D*C**2,kernel_size,stride,bias=bias,pad_type=pad_type,norm_type=_A,act_type=_A,convtype=convtype);F=nn.PixelShuffle(C);G=norm(B,D)if B else _A;H=act(A)if A else _A;return sequential(E,F,G,H)
def upconv_block(in_nc,out_nc,upscale_factor=2,kernel_size=3,stride=1,bias=_D,pad_type=_C,norm_type=_A,act_type=_H,mode=_J,convtype=_G):' Upconv layer ';B=convtype;A=upscale_factor;A=(1,A,A)if B==_L else A;C=Upsample(scale_factor=A,mode=mode);D=conv_block(in_nc,out_nc,kernel_size,stride,bias=bias,pad_type=pad_type,norm_type=norm_type,act_type=act_type,convtype=B);return sequential(C,D)
def make_layer(basic_block,num_basic_block,**B):
	'Make layers by stacking the same blocks.\n    Args:\n        basic_block (nn.module): nn.module class for basic block. (block)\n        num_basic_block (int): number of blocks. (n_layers)\n    Returns:\n        nn.Sequential: Stacked blocks in nn.Sequential.\n    ';A=[]
	for C in range(num_basic_block):A.append(basic_block(**B))
	return nn.Sequential(*A)
def act(act_type,inplace=_D,neg_slope=.2,n_prelu=1,beta=1.):
	' activation helper ';C=neg_slope;D=inplace;A=act_type;A=A.lower()
	if A==_H:B=nn.ReLU(D)
	elif A in(_F,'lrelu'):B=nn.LeakyReLU(C,D)
	elif A==_I:B=nn.PReLU(num_parameters=n_prelu,init=C)
	elif A=='tanh':B=nn.Tanh()
	elif A=='sigmoid':B=nn.Sigmoid()
	else:raise NotImplementedError(f"activation layer [{A}] is not found")
	return B
class Identity(nn.Module):
	def __init__(A,*B):super(Identity,A).__init__()
	def forward(A,x,*B):return x
def norm(norm_type,nc):
	' Return a normalization layer ';A=norm_type;A=A.lower()
	if A=='batch':B=nn.BatchNorm2d(nc,affine=_D)
	elif A=='instance':B=nn.InstanceNorm2d(nc,affine=_B)
	elif A=='none':
		def C(x):return Identity()
	else:raise NotImplementedError(f"normalization layer [{A}] is not found")
	return B
def pad(pad_type,padding):
	' padding layer helper ';B=padding;A=pad_type;A=A.lower()
	if B==0:return
	if A=='reflect':C=nn.ReflectionPad2d(B)
	elif A=='replicate':C=nn.ReplicationPad2d(B)
	elif A==_C:C=nn.ZeroPad2d(B)
	else:raise NotImplementedError(f"padding layer [{A}] is not implemented")
	return C
def get_valid_padding(kernel_size,dilation):A=kernel_size;A=A+(A-1)*(dilation-1);B=(A-1)//2;return B
class ShortcutBlock(nn.Module):
	' Elementwise sum the output of a submodule to its input '
	def __init__(A,submodule):super(ShortcutBlock,A).__init__();A.sub=submodule
	def forward(A,x):B=x+A.sub(x);return B
	def __repr__(A):return'Identity + \n|'+A.sub.__repr__().replace('\n','\n|')
def sequential(*A):
	' Flatten Sequential. It unwraps nn.Sequential. '
	if len(A)==1:
		if isinstance(A[0],OrderedDict):raise NotImplementedError('sequential does not support OrderedDict input.')
		return A[0]
	C=[]
	for B in A:
		if isinstance(B,nn.Sequential):
			for D in B.children():C.append(D)
		elif isinstance(B,nn.Module):C.append(B)
	return nn.Sequential(*C)
def conv_block(in_nc,out_nc,kernel_size,stride=1,dilation=1,groups=1,bias=_D,pad_type=_C,norm_type=_A,act_type=_H,mode=_E,convtype=_G,spectral_norm=_B):
	' Conv layer with padding, normalization, activation ';Q='NAC';N=convtype;H=mode;I=act_type;J=pad_type;K=bias;L=groups;M=stride;C=norm_type;D=dilation;E=kernel_size;F=out_nc;G=in_nc;assert H in[_E,Q,'CNAC'],f"Wrong conv mode [{H}]";A=get_valid_padding(E,D);R=pad(J,A)if J and J!=_C else _A;A=A if J==_C else 0
	if N=='PartialConv2D':from torchvision.ops import PartialConv2d as S;B=S(G,F,kernel_size=E,stride=M,padding=A,dilation=D,bias=K,groups=L)
	elif N=='DeformConv2D':from torchvision.ops import DeformConv2d as T;B=T(G,F,kernel_size=E,stride=M,padding=A,dilation=D,bias=K,groups=L)
	elif N==_L:B=nn.Conv3d(G,F,kernel_size=E,stride=M,padding=A,dilation=D,bias=K,groups=L)
	else:B=nn.Conv2d(G,F,kernel_size=E,stride=M,padding=A,dilation=D,bias=K,groups=L)
	if spectral_norm:B=nn.utils.spectral_norm(B)
	O=act(I)if I else _A
	if _E in H:P=norm(C,F)if C else _A;return sequential(R,B,P,O)
	elif H==Q:
		if C is _A and I is not _A:O=act(I,inplace=_B)
		P=norm(C,G)if C else _A;return sequential(P,O,R,B)