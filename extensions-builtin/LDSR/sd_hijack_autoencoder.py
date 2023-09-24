_D='colorize'
_C=False
_B=True
_A=None
import numpy as np,torch,pytorch_lightning as pl,torch.nn.functional as F
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from ldm.modules.ema import LitEma
from vqvae_quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder,Decoder
from ldm.util import instantiate_from_config
import ldm.models.autoencoder
from packaging import version
class VQModel(pl.LightningModule):
	def __init__(A,ddconfig,lossconfig,n_embed,embed_dim,ckpt_path=_A,ignore_keys=_A,image_key='image',colorize_nlabels=_A,monitor=_A,batch_resize_range=_A,scheduler_config=_A,lr_g_factor=1.,remap=_A,sane_index_shape=_C,use_ema=_C):
		I='z_channels';H=batch_resize_range;G=monitor;F=ckpt_path;E=n_embed;D=colorize_nlabels;C=embed_dim;B=ddconfig;super().__init__();A.embed_dim=C;A.n_embed=E;A.image_key=image_key;A.encoder=Encoder(**B);A.decoder=Decoder(**B);A.loss=instantiate_from_config(lossconfig);A.quantize=VectorQuantizer(E,C,beta=.25,remap=remap,sane_index_shape=sane_index_shape);A.quant_conv=torch.nn.Conv2d(B[I],C,1);A.post_quant_conv=torch.nn.Conv2d(C,B[I],1)
		if D is not _A:assert type(D)==int;A.register_buffer(_D,torch.randn(3,D,1,1))
		if G is not _A:A.monitor=G
		A.batch_resize_range=H
		if A.batch_resize_range is not _A:print(f"{A.__class__.__name__}: Using per-batch resizing in range {H}.")
		A.use_ema=use_ema
		if A.use_ema:A.model_ema=LitEma(A);print(f"Keeping EMAs of {len(list(A.model_ema.buffers()))}.")
		if F is not _A:A.init_from_ckpt(F,ignore_keys=ignore_keys or[])
		A.scheduler_config=scheduler_config;A.lr_g_factor=lr_g_factor
	@contextmanager
	def ema_scope(self,context=_A):
		B=context;A=self
		if A.use_ema:
			A.model_ema.store(A.parameters());A.model_ema.copy_to(A)
			if B is not _A:print(f"{B}: Switched to EMA weights")
		try:yield _A
		finally:
			if A.use_ema:
				A.model_ema.restore(A.parameters())
				if B is not _A:print(f"{B}: Restored training weights")
	def init_from_ckpt(E,path,ignore_keys=_A):
		A=torch.load(path,map_location='cpu')['state_dict'];F=list(A.keys())
		for B in F:
			for G in ignore_keys or[]:
				if B.startswith(G):print('Deleting key {} from state_dict.'.format(B));del A[B]
		C,D=E.load_state_dict(A,strict=_C);print(f"Restored from {path} with {len(C)} missing and {len(D)} unexpected keys")
		if C:print(f"Missing Keys: {C}")
		if D:print(f"Unexpected Keys: {D}")
	def on_train_batch_end(A,*B,**C):
		if A.use_ema:A.model_ema(A)
	def encode(A,x):B=A.encoder(x);B=A.quant_conv(B);C,D,E=A.quantize(B);return C,D,E
	def encode_to_prequant(B,x):A=B.encoder(x);A=B.quant_conv(A);return A
	def decode(B,quant):A=quant;A=B.post_quant_conv(A);C=B.decoder(A);return C
	def decode_code(A,code_b):B=A.quantize.embed_code(code_b);C=A.decode(B);return C
	def forward(A,input,return_pred_indices=_C):
		D,B,(E,E,F)=A.encode(input);C=A.decode(D)
		if return_pred_indices:return C,B,F
		return C,B
	def get_input(B,batch,k):
		A=batch[k]
		if len(A.shape)==3:A=A[...,_A]
		A=A.permute(0,3,1,2).to(memory_format=torch.contiguous_format).float()
		if B.batch_resize_range is not _A:
			E=B.batch_resize_range[0];D=B.batch_resize_range[1]
			if B.global_step<=4:C=D
			else:C=np.random.choice(np.arange(E,D+16,16))
			if C!=A.shape[2]:A=F.interpolate(A,size=C,mode='bicubic')
			A=A.detach()
		return A
	def training_step(A,batch,batch_idx,optimizer_idx):
		F='train';B=optimizer_idx;C=A.get_input(batch,A.image_key);D,E,G=A(C,return_pred_indices=_B)
		if B==0:H,I=A.loss(E,C,D,B,A.global_step,last_layer=A.get_last_layer(),split=F,predicted_indices=G);A.log_dict(I,prog_bar=_C,logger=_B,on_step=_B,on_epoch=_B);return H
		if B==1:J,K=A.loss(E,C,D,B,A.global_step,last_layer=A.get_last_layer(),split=F);A.log_dict(K,prog_bar=_C,logger=_B,on_step=_B,on_epoch=_B);return J
	def validation_step(A,batch,batch_idx):
		C=batch_idx;B=batch;D=A._validation_step(B,C)
		with A.ema_scope():A._validation_step(B,C,suffix='_ema')
		return D
	def _validation_step(A,batch,batch_idx,suffix=''):
		H='val';B=suffix;C=A.get_input(batch,A.image_key);E,F,G=A(C,return_pred_indices=_B);I,D=A.loss(F,C,E,0,A.global_step,last_layer=A.get_last_layer(),split=H+B,predicted_indices=G);L,J=A.loss(F,C,E,1,A.global_step,last_layer=A.get_last_layer(),split=H+B,predicted_indices=G);K=D[f"val{B}/rec_loss"];A.log(f"val{B}/rec_loss",K,prog_bar=_B,logger=_B,on_step=_C,on_epoch=_B,sync_dist=_B);A.log(f"val{B}/aeloss",I,prog_bar=_B,logger=_B,on_step=_C,on_epoch=_B,sync_dist=_B)
		if version.parse(pl.__version__)>=version.parse('1.4.0'):del D[f"val{B}/rec_loss"]
		A.log_dict(D);A.log_dict(J);return A.log_dict
	def configure_optimizers(A):
		J='step';I='frequency';H='interval';G='scheduler';E=A.learning_rate;F=A.lr_g_factor*A.learning_rate;print('lr_d',E);print('lr_g',F);C=torch.optim.Adam(list(A.encoder.parameters())+list(A.decoder.parameters())+list(A.quantize.parameters())+list(A.quant_conv.parameters())+list(A.post_quant_conv.parameters()),lr=F,betas=(.5,.9));D=torch.optim.Adam(A.loss.discriminator.parameters(),lr=E,betas=(.5,.9))
		if A.scheduler_config is not _A:B=instantiate_from_config(A.scheduler_config);print('Setting up LambdaLR scheduler...');B=[{G:LambdaLR(C,lr_lambda=B.schedule),H:J,I:1},{G:LambdaLR(D,lr_lambda=B.schedule),H:J,I:1}];return[C,D],B
		return[C,D],[]
	def get_last_layer(A):return A.decoder.conv_out.weight
	def log_images(B,batch,only_inputs=_C,plot_ema=_C,**H):
		F='inputs';C={};A=B.get_input(batch,B.image_key);A=A.to(B.device)
		if only_inputs:C[F]=A;return C
		D,G=B(A)
		if A.shape[1]>3:assert D.shape[1]>3;A=B.to_rgb(A);D=B.to_rgb(D)
		C[F]=A;C['reconstructions']=D
		if plot_ema:
			with B.ema_scope():
				E,G=B(A)
				if A.shape[1]>3:E=B.to_rgb(E)
				C['reconstructions_ema']=E
		return C
	def to_rgb(A,x):
		assert A.image_key=='segmentation'
		if not hasattr(A,_D):A.register_buffer(_D,torch.randn(3,x.shape[1],1,1).to(x))
		x=F.conv2d(x,weight=A.colorize);x=2.*(x-x.min())/(x.max()-x.min())-1.;return x
class VQModelInterface(VQModel):
	def __init__(B,embed_dim,*C,**D):A=embed_dim;super().__init__(*C,embed_dim=A,**D);B.embed_dim=A
	def encode(B,x):A=B.encoder(x);A=B.quant_conv(A);return A
	def decode(B,h,force_not_quantize=_C):
		if not force_not_quantize:A,D,E=B.quantize(h)
		else:A=h
		A=B.post_quant_conv(A);C=B.decoder(A);return C
ldm.models.autoencoder.VQModel=VQModel
ldm.models.autoencoder.VQModelInterface=VQModelInterface