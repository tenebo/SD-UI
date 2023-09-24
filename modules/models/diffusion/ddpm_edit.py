'\nwild mixture of\nhttps://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py\nhttps://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py\nhttps://github.com/CompVis/taming-transformers\n-- merci\n'
_l='Progressive Generation'
_k='coordinates_bbox'
_j='b h w c -> b c h w'
_i='scale_factor'
_h='denoise_row'
_g='samples'
_f='Plotting'
_e='diffusion_row'
_d='1 -> b'
_c='inputs'
_b='Sampling t'
_a='original_image_size'
_Z='reducing stride'
_Y='reducing Kernel'
_X='vqf'
_W='patch_distributed_vq'
_V='__is_unconditional__'
_U='b n c h w -> (b n) c h w'
_T='n b c h w -> b n c h w'
_S='train'
_R='image'
_Q='linear'
_P='adm'
_O='hybrid'
_N='stride'
_M='ks'
_L='split_input_params'
_K='crossattn'
_J='concat'
_I='x0'
_H='c_concat'
_G='eps'
_F='c_crossattn'
_E=.0
_D=1.
_C=True
_B=False
_A=None
import torch,torch.nn as nn,numpy as np,pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange,repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.util import log_txt_as_img,exists,default,ismap,isimage,mean_flat,count_params,instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl,DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface,IdentityFirstStage,AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule,extract_into_tensor,noise_like
from ldm.models.diffusion.ddim import DDIMSampler
__conditioning_keys__={_J:_H,_K:_F,_P:'y'}
def disabled_train(self,mode=_C):'Overwrite model.train with this function to make sure train/eval mode\n    does not change anymore.';return self
def uniform_on_device(r1,r2,shape,device):return(r1-r2)*torch.rand(*shape,device=device)+r2
class DDPM(pl.LightningModule):
	def __init__(A,unet_config,timesteps=1000,beta_schedule=_Q,loss_type='l2',ckpt_path=_A,ignore_keys=_A,load_only_unet=_B,monitor='val/loss',use_ema=_C,first_stage_key=_R,image_size=256,channels=3,log_every_t=100,clip_denoised=_C,linear_start=.0001,linear_end=.02,cosine_s=.008,given_betas=_A,original_elbo_weight=_E,v_posterior=_E,l_simple_weight=_D,conditioning_key=_A,parameterization=_G,scheduler_config=_A,use_positional_encodings=_B,learn_logvar=_B,logvar_init=_E,load_ema=_C):
		F=load_ema;E=scheduler_config;D=parameterization;C=monitor;B=ckpt_path;super().__init__();assert D in[_G,_I],'currently only supporting "eps" and "x0"';A.parameterization=D;print(f"{A.__class__.__name__}: Running in {A.parameterization}-prediction mode");A.cond_stage_model=_A;A.clip_denoised=clip_denoised;A.log_every_t=log_every_t;A.first_stage_key=first_stage_key;A.image_size=image_size;A.channels=channels;A.use_positional_encodings=use_positional_encodings;A.model=DiffusionWrapper(unet_config,conditioning_key);count_params(A.model,verbose=_C);A.use_ema=use_ema;A.use_scheduler=E is not _A
		if A.use_scheduler:A.scheduler_config=E
		A.v_posterior=v_posterior;A.original_elbo_weight=original_elbo_weight;A.l_simple_weight=l_simple_weight
		if C is not _A:A.monitor=C
		if A.use_ema and F:A.model_ema=LitEma(A.model);print(f"Keeping EMAs of {len(list(A.model_ema.buffers()))}.")
		if B is not _A:
			A.init_from_ckpt(B,ignore_keys=ignore_keys or[],only_model=load_only_unet)
			if A.use_ema and not F:A.model_ema=LitEma(A.model);print(f"Keeping EMAs of {len(list(A.model_ema.buffers()))}.")
		A.register_schedule(given_betas=given_betas,beta_schedule=beta_schedule,timesteps=timesteps,linear_start=linear_start,linear_end=linear_end,cosine_s=cosine_s);A.loss_type=loss_type;A.learn_logvar=learn_logvar;A.logvar=torch.full(fill_value=logvar_init,size=(A.num_timesteps,))
		if A.learn_logvar:A.logvar=nn.Parameter(A.logvar,requires_grad=_C)
	def register_schedule(A,given_betas=_A,beta_schedule=_Q,timesteps=1000,linear_start=.0001,linear_end=.02,cosine_s=.008):
		K=linear_end;J=linear_start;I=given_betas;G=timesteps
		if exists(I):D=I
		else:D=make_beta_schedule(beta_schedule,G,linear_start=J,linear_end=K,cosine_s=cosine_s)
		H=_D-D;B=np.cumprod(H,axis=0);E=np.append(_D,B[:-1]);G,=D.shape;A.num_timesteps=int(G);A.linear_start=J;A.linear_end=K;assert B.shape[0]==A.num_timesteps,'alphas have to be defined for each timestep';C=partial(torch.tensor,dtype=torch.float32);A.register_buffer('betas',C(D));A.register_buffer('alphas_cumprod',C(B));A.register_buffer('alphas_cumprod_prev',C(E));A.register_buffer('sqrt_alphas_cumprod',C(np.sqrt(B)));A.register_buffer('sqrt_one_minus_alphas_cumprod',C(np.sqrt(_D-B)));A.register_buffer('log_one_minus_alphas_cumprod',C(np.log(_D-B)));A.register_buffer('sqrt_recip_alphas_cumprod',C(np.sqrt(_D/B)));A.register_buffer('sqrt_recipm1_alphas_cumprod',C(np.sqrt(_D/B-1)));L=(1-A.v_posterior)*D*(_D-E)/(_D-B)+A.v_posterior*D;A.register_buffer('posterior_variance',C(L));A.register_buffer('posterior_log_variance_clipped',C(np.log(np.maximum(L,1e-20))));A.register_buffer('posterior_mean_coef1',C(D*np.sqrt(E)/(_D-B)));A.register_buffer('posterior_mean_coef2',C((_D-E)*np.sqrt(H)/(_D-B)))
		if A.parameterization==_G:F=A.betas**2/(2*A.posterior_variance*C(H)*(1-A.alphas_cumprod))
		elif A.parameterization==_I:F=.5*np.sqrt(torch.Tensor(B))/(2.*1-torch.Tensor(B))
		else:raise NotImplementedError('mu not supported')
		F[0]=F[1];A.register_buffer('lvlb_weights',F,persistent=_B);assert not torch.isnan(A.lvlb_weights).all()
	@contextmanager
	def ema_scope(self,context=_A):
		B=context;A=self
		if A.use_ema:
			A.model_ema.store(A.model.parameters());A.model_ema.copy_to(A.model)
			if B is not _A:print(f"{B}: Switched to EMA weights")
		try:yield _A
		finally:
			if A.use_ema:
				A.model_ema.restore(A.model.parameters())
				if B is not _A:print(f"{B}: Restored training weights")
	def init_from_ckpt(D,path,ignore_keys=_A,only_model=_B):
		J='state_dict';C=ignore_keys;C=C or[];A=torch.load(path,map_location='cpu')
		if J in list(A.keys()):A=A[J]
		K=list(A.keys());L=['model.diffusion_model.input_blocks.0.0.weight','model_ema.diffusion_modelinput_blocks00weight'];I=D.state_dict()
		for B in L:
			if B not in A or B not in I:continue
			E=I[B]
			if E.size()!=A[B].size():print(f"Manual init: {B}");E.zero_();E[:,:4,:,:].copy_(A[B]);C.append(B)
		for F in K:
			for M in C:
				if F.startswith(M):print(f"Deleting key {F} from state_dict.");del A[F]
		G,H=D.load_state_dict(A,strict=_B)if not only_model else D.model.load_state_dict(A,strict=_B);print(f"Restored from {path} with {len(G)} missing and {len(H)} unexpected keys")
		if G:print(f"Missing Keys: {G}")
		if H:print(f"Unexpected Keys: {H}")
	def q_mean_variance(B,x_start,t):"\n        Get the distribution q(x_t | x_0).\n        :param x_start: the [N x C x ...] tensor of noiseless inputs.\n        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.\n        :return: A tuple (mean, variance, log_variance), all of x_start's shape.\n        ";A=x_start;C=extract_into_tensor(B.sqrt_alphas_cumprod,t,A.shape)*A;D=extract_into_tensor(_D-B.alphas_cumprod,t,A.shape);E=extract_into_tensor(B.log_one_minus_alphas_cumprod,t,A.shape);return C,D,E
	def predict_start_from_noise(B,x_t,t,noise):A=x_t;return extract_into_tensor(B.sqrt_recip_alphas_cumprod,t,A.shape)*A-extract_into_tensor(B.sqrt_recipm1_alphas_cumprod,t,A.shape)*noise
	def q_posterior(B,x_start,x_t,t):A=x_t;C=extract_into_tensor(B.posterior_mean_coef1,t,A.shape)*x_start+extract_into_tensor(B.posterior_mean_coef2,t,A.shape)*A;D=extract_into_tensor(B.posterior_variance,t,A.shape);E=extract_into_tensor(B.posterior_log_variance_clipped,t,A.shape);return C,D,E
	def p_mean_variance(A,x,t,clip_denoised):
		C=A.model(x,t)
		if A.parameterization==_G:B=A.predict_start_from_noise(x,t=t,noise=C)
		elif A.parameterization==_I:B=C
		if clip_denoised:B.clamp_(-_D,_D)
		D,E,F=A.q_posterior(x_start=B,x_t=x,t=t);return D,E,F
	@torch.no_grad()
	def p_sample(self,x,t,clip_denoised=_C,repeat_noise=_B):A,*B,C=*x.shape,x.device;D,B,E=self.p_mean_variance(x=x,t=t,clip_denoised=clip_denoised);F=noise_like(x.shape,C,repeat_noise);G=(1-(t==0).float()).reshape(A,*(1,)*(len(x.shape)-1));return D+G*(.5*E).exp()*F
	@torch.no_grad()
	def p_sample_loop(self,shape,return_intermediates=_B):
		D=shape;A=self;E=A.betas.device;G=D[0];B=torch.randn(D,device=E);F=[B]
		for C in tqdm(reversed(range(0,A.num_timesteps)),desc=_b,total=A.num_timesteps):
			B=A.p_sample(B,torch.full((G,),C,device=E,dtype=torch.long),clip_denoised=A.clip_denoised)
			if C%A.log_every_t==0 or C==A.num_timesteps-1:F.append(B)
		if return_intermediates:return B,F
		return B
	@torch.no_grad()
	def sample(self,batch_size=16,return_intermediates=_B):A=self;B=A.image_size;C=A.channels;return A.p_sample_loop((batch_size,C,B,B),return_intermediates=return_intermediates)
	def q_sample(C,x_start,t,noise=_A):B=noise;A=x_start;B=default(B,lambda:torch.randn_like(A));return extract_into_tensor(C.sqrt_alphas_cumprod,t,A.shape)*A+extract_into_tensor(C.sqrt_one_minus_alphas_cumprod,t,A.shape)*B
	def get_loss(D,pred,target,mean=_C):
		C=target;B=pred
		if D.loss_type=='l1':
			A=(C-B).abs()
			if mean:A=A.mean()
		elif D.loss_type=='l2':
			if mean:A=torch.nn.functional.mse_loss(C,B)
			else:A=torch.nn.functional.mse_loss(C,B,reduction='none')
		else:raise NotImplementedError("unknown loss type '{loss_type}'")
		return A
	def p_losses(A,x_start,t,noise=_A):
		E=x_start;C=noise;C=default(C,lambda:torch.randn_like(E));I=A.q_sample(x_start=E,t=t,noise=C);J=A.model(I,t);D={}
		if A.parameterization==_G:G=C
		elif A.parameterization==_I:G=E
		else:raise NotImplementedError(f"Paramterization {A.parameterization} not yet supported")
		B=A.get_loss(J,G,mean=_B).mean(dim=[1,2,3]);F=_S if A.training else'val';D.update({f"{F}/loss_simple":B.mean()});K=B.mean()*A.l_simple_weight;H=(A.lvlb_weights[t]*B).mean();D.update({f"{F}/loss_vlb":H});B=K+A.original_elbo_weight*H;D.update({f"{F}/loss":B});return B,D
	def forward(A,x,*B,**C):D=torch.randint(0,A.num_timesteps,(x.shape[0],),device=A.device).long();return A.p_losses(x,D,*B,**C)
	def get_input(A,batch,k):return batch[k]
	def shared_step(A,batch):B=A.get_input(batch,A.first_stage_key);C,D=A(B);return C,D
	def training_step(A,batch,batch_idx):
		B,C=A.shared_step(batch);A.log_dict(C,prog_bar=_C,logger=_C,on_step=_C,on_epoch=_C);A.log('global_step',A.global_step,prog_bar=_C,logger=_C,on_step=_C,on_epoch=_B)
		if A.use_scheduler:D=A.optimizers().param_groups[0]['lr'];A.log('lr_abs',D,prog_bar=_C,logger=_C,on_step=_C,on_epoch=_B)
		return B
	@torch.no_grad()
	def validation_step(self,batch,batch_idx):
		C=batch;A=self;D,E=A.shared_step(C)
		with A.ema_scope():D,B=A.shared_step(C);B={f"{A}_ema":B[A]for A in B}
		A.log_dict(E,prog_bar=_B,logger=_C,on_step=_B,on_epoch=_C);A.log_dict(B,prog_bar=_B,logger=_C,on_step=_B,on_epoch=_C)
	def on_train_batch_end(A,*B,**C):
		if A.use_ema:A.model_ema(A.model)
	def _get_rows_from_list(D,samples):B=samples;C=len(B);A=rearrange(B,_T);A=rearrange(A,_U);A=make_grid(A,nrow=C);return A
	@torch.no_grad()
	def log_images(self,batch,N=8,n_row=2,sample=_C,return_keys=_A,**M):
		F=return_keys;E=n_row;A=self;B={};D=A.get_input(batch,A.first_stage_key);N=min(D.shape[0],N);E=min(D.shape[0],E);D=D.to(A.device)[:N];B[_c]=D;G=[];H=D[:E]
		for C in range(A.num_timesteps):
			if C%A.log_every_t==0 or C==A.num_timesteps-1:C=repeat(torch.tensor([C]),_d,b=E);C=C.to(A.device).long();I=torch.randn_like(H);J=A.q_sample(x_start=H,t=C,noise=I);G.append(J)
		B[_e]=A._get_rows_from_list(G)
		if sample:
			with A.ema_scope(_f):K,L=A.sample(batch_size=N,return_intermediates=_C)
			B[_g]=K;B[_h]=A._get_rows_from_list(L)
		if F:
			if np.intersect1d(list(B.keys()),F).shape[0]==0:return B
			else:return{A:B[A]for A in F}
		return B
	def configure_optimizers(A):
		C=A.learning_rate;B=list(A.model.parameters())
		if A.learn_logvar:B=B+[A.logvar]
		D=torch.optim.AdamW(B,lr=C);return D
class LatentDiffusion(DDPM):
	'main class'
	def __init__(A,first_stage_config,cond_stage_config,num_timesteps_cond=_A,cond_stage_key=_R,cond_stage_trainable=_B,concat_mode=_C,cond_stage_forward=_A,conditioning_key=_A,scale_factor=_D,scale_by_std=_B,load_ema=_C,*K,**C):
		I=load_ema;H=scale_by_std;G=scale_factor;F=concat_mode;E=cond_stage_config;D=first_stage_config;B=conditioning_key;A.num_timesteps_cond=default(num_timesteps_cond,1);A.scale_by_std=H;assert A.num_timesteps_cond<=C['timesteps']
		if B is _A:B=_J if F else _K
		if E==_V:B=_A
		J=C.pop('ckpt_path',_A);L=C.pop('ignore_keys',[]);super().__init__(*K,conditioning_key=B,load_ema=I,**C);A.concat_mode=F;A.cond_stage_trainable=cond_stage_trainable;A.cond_stage_key=cond_stage_key
		try:A.num_downs=len(D.params.ddconfig.ch_mult)-1
		except Exception:A.num_downs=0
		if not H:A.scale_factor=G
		else:A.register_buffer(_i,torch.tensor(G))
		A.instantiate_first_stage(D);A.instantiate_cond_stage(E);A.cond_stage_forward=cond_stage_forward;A.clip_denoised=_B;A.bbox_tokenizer=_A;A.restarted_from_ckpt=_B
		if J is not _A:
			A.init_from_ckpt(J,L);A.restarted_from_ckpt=_C
			if A.use_ema and not I:A.model_ema=LitEma(A.model);print(f"Keeping EMAs of {len(list(A.model_ema.buffers()))}.")
	def make_cond_schedule(A):A.cond_ids=torch.full(size=(A.num_timesteps,),fill_value=A.num_timesteps-1,dtype=torch.long);B=torch.round(torch.linspace(0,A.num_timesteps-1,A.num_timesteps_cond)).long();A.cond_ids[:A.num_timesteps_cond]=B
	@rank_zero_only
	@torch.no_grad()
	def on_train_batch_start(self,batch,batch_idx,dataloader_idx):
		C='### USING STD-RESCALING ###';A=self
		if A.scale_by_std and A.current_epoch==0 and A.global_step==0 and batch_idx==0 and not A.restarted_from_ckpt:assert A.scale_factor==_D,'rather not use custom rescaling and std-rescaling simultaneously';print(C);B=super().get_input(batch,A.first_stage_key);B=B.to(A.device);D=A.encode_first_stage(B);E=A.get_first_stage_encoding(D).detach();del A.scale_factor;A.register_buffer(_i,_D/E.flatten().std());print(f"setting self.scale_factor to {A.scale_factor}");print(C)
	def register_schedule(A,given_betas=_A,beta_schedule=_Q,timesteps=1000,linear_start=.0001,linear_end=.02,cosine_s=.008):
		super().register_schedule(given_betas,beta_schedule,timesteps,linear_start,linear_end,cosine_s);A.shorten_cond_schedule=A.num_timesteps_cond>1
		if A.shorten_cond_schedule:A.make_cond_schedule()
	def instantiate_first_stage(A,config):
		B=instantiate_from_config(config);A.first_stage_model=B.eval();A.first_stage_model.train=disabled_train
		for C in A.first_stage_model.parameters():C.requires_grad=_B
	def instantiate_cond_stage(A,config):
		D='__is_first_stage__';B=config
		if not A.cond_stage_trainable:
			if B==D:print('Using first stage also as cond stage.');A.cond_stage_model=A.first_stage_model
			elif B==_V:print(f"Training {A.__class__.__name__} as an unconditional model.");A.cond_stage_model=_A
			else:
				C=instantiate_from_config(B);A.cond_stage_model=C.eval();A.cond_stage_model.train=disabled_train
				for E in A.cond_stage_model.parameters():E.requires_grad=_B
		else:assert B!=D;assert B!=_V;C=instantiate_from_config(B);A.cond_stage_model=C
	def _get_denoise_row_from_list(C,samples,desc='',force_no_decoder_quantization=_B):
		A=[]
		for D in tqdm(samples,desc=desc):A.append(C.decode_first_stage(D.to(C.device),force_not_quantize=force_no_decoder_quantization))
		E=len(A);A=torch.stack(A);B=rearrange(A,_T);B=rearrange(B,_U);B=make_grid(B,nrow=E);return B
	def get_first_stage_encoding(C,encoder_posterior):
		A=encoder_posterior
		if isinstance(A,DiagonalGaussianDistribution):B=A.sample()
		elif isinstance(A,torch.Tensor):B=A
		else:raise NotImplementedError(f"encoder_posterior of type '{type(A)}' not yet implemented")
		return C.scale_factor*B
	def get_learned_conditioning(A,c):
		if A.cond_stage_forward is _A:
			if hasattr(A.cond_stage_model,'encode')and callable(A.cond_stage_model.encode):
				c=A.cond_stage_model.encode(c)
				if isinstance(c,DiagonalGaussianDistribution):c=c.mode()
			else:c=A.cond_stage_model(c)
		else:assert hasattr(A.cond_stage_model,A.cond_stage_forward);c=getattr(A.cond_stage_model,A.cond_stage_forward)(c)
		return c
	def meshgrid(D,h,w):A=torch.arange(0,h).view(h,1,1).repeat(1,w,1);B=torch.arange(0,w).view(1,w,1).repeat(h,1,1);C=torch.cat([A,B],dim=-1);return C
	def delta_border(B,h,w):'\n        :param h: height\n        :param w: width\n        :return: normalized distance to image border,\n         wtith min distance = 0 at border and max dist = 0.5 at image center\n        ';C=torch.tensor([h-1,w-1]).view(1,1,2);A=B.meshgrid(h,w)/C;D=torch.min(A,dim=-1,keepdims=_C)[0];E=torch.min(1-A,dim=-1,keepdims=_C)[0];F=torch.min(torch.cat([D,E],dim=-1),dim=-1)[0];return F
	def get_weighting(A,h,w,Ly,Lx,device):
		D=device;B=A.delta_border(h,w);B=torch.clip(B,A.split_input_params['clip_min_weight'],A.split_input_params['clip_max_weight']);B=B.view(1,h*w,1).repeat(1,1,Ly*Lx).to(D)
		if A.split_input_params['tie_braker']:C=A.delta_border(Ly,Lx);C=torch.clip(C,A.split_input_params['clip_min_tie_weight'],A.split_input_params['clip_max_tie_weight']);C=C.view(1,1,Ly*Lx).to(D);B=B*C
		return B
	def get_fold_unfold(L,x,kernel_size,stride,uf=1,df=1):
		'\n        :param x: img of size (bs, c, h, w)\n        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])\n        ';E=stride;C=df;B=uf;A=kernel_size;P,Q,J,K=x.shape;F=(J-A[0])//E[0]+1;G=(K-A[1])//E[1]+1
		if B==1 and C==1:H=dict(kernel_size=A,dilation=1,padding=0,stride=E);M=torch.nn.Unfold(**H);I=torch.nn.Fold(output_size=x.shape[2:],**H);D=L.get_weighting(A[0],A[1],F,G,x.device).to(x.dtype);N=I(D).view(1,1,J,K);D=D.view((1,1,A[0],A[1],F*G))
		elif B>1 and C==1:H=dict(kernel_size=A,dilation=1,padding=0,stride=E);M=torch.nn.Unfold(**H);O=dict(kernel_size=(A[0]*B,A[0]*B),dilation=1,padding=0,stride=(E[0]*B,E[1]*B));I=torch.nn.Fold(output_size=(x.shape[2]*B,x.shape[3]*B),**O);D=L.get_weighting(A[0]*B,A[1]*B,F,G,x.device).to(x.dtype);N=I(D).view(1,1,J*B,K*B);D=D.view((1,1,A[0]*B,A[1]*B,F*G))
		elif C>1 and B==1:H=dict(kernel_size=A,dilation=1,padding=0,stride=E);M=torch.nn.Unfold(**H);O=dict(kernel_size=(A[0]//C,A[0]//C),dilation=1,padding=0,stride=(E[0]//C,E[1]//C));I=torch.nn.Fold(output_size=(x.shape[2]//C,x.shape[3]//C),**O);D=L.get_weighting(A[0]//C,A[1]//C,F,G,x.device).to(x.dtype);N=I(D).view(1,1,J//C,K//C);D=D.view((1,1,A[0]//C,A[1]//C,F*G))
		else:raise NotImplementedError
		return I,M,N,D
	@torch.no_grad()
	def get_input(self,batch,k,return_first_stage_outputs=_B,force_c_encode=_B,cond_key=_A,return_original_cond=_B,bs=_A,uncond=.05):
		J=batch;F=uncond;E=cond_key;D=bs;A=self;B=super().get_input(J,k)
		if D is not _A:B=B[:D]
		B=B.to(A.device);L=A.encode_first_stage(B);K=A.get_first_stage_encoding(L).detach();E=E or A.cond_stage_key;C=super().get_input(J,E)
		if D is not _A:C[_F]=C[_F][:D];C[_H]=C[_H][:D]
		G={};H=torch.rand(B.size(0),device=B.device);M=rearrange(H<2*F,'n -> n 1 1');N=1-rearrange((H>=F).float()*(H<3*F).float(),'n -> n 1 1 1');O=A.get_learned_conditioning(['']);G[_F]=[torch.where(M,O,A.get_learned_conditioning(C[_F]).detach())];G[_H]=[N*A.encode_first_stage(C[_H].to(A.device)).mode().detach()];I=[K,G]
		if return_first_stage_outputs:P=A.decode_first_stage(K);I.extend([B,P])
		if return_original_cond:I.append(C)
		return I
	@torch.no_grad()
	def decode_first_stage(self,z,predict_cids=_B,force_not_quantize=_B):
		H=force_not_quantize;E=predict_cids;A=self
		if E:
			if z.dim()==4:z=torch.argmax(z.exp(),dim=1).long()
			z=A.first_stage_model.quantize.get_codebook_entry(z,shape=_A);z=rearrange(z,_j).contiguous()
		z=_D/A.scale_factor*z
		if hasattr(A,_L):
			if A.split_input_params[_W]:
				B=A.split_input_params[_M];D=A.split_input_params[_N];K=A.split_input_params[_X];P,Q,F,G=z.shape
				if B[0]>F or B[1]>G:B=min(B[0],F),min(B[1],G);print(_Y)
				if D[0]>F or D[1]>G:D=min(D[0],F),min(D[1],G);print(_Z)
				L,M,N,O=A.get_fold_unfold(z,B,D,uf=K);z=M(z);z=z.view((z.shape[0],-1,B[0],B[1],z.shape[-1]))
				if isinstance(A.first_stage_model,VQModelInterface):J=[A.first_stage_model.decode(z[:,:,:,:,B],force_not_quantize=E or H)for B in range(z.shape[-1])]
				else:J=[A.first_stage_model.decode(z[:,:,:,:,B])for B in range(z.shape[-1])]
				C=torch.stack(J,axis=-1);C=C*O;C=C.view((C.shape[0],-1,C.shape[-1]));I=L(C);I=I/N;return I
			elif isinstance(A.first_stage_model,VQModelInterface):return A.first_stage_model.decode(z,force_not_quantize=E or H)
			else:return A.first_stage_model.decode(z)
		elif isinstance(A.first_stage_model,VQModelInterface):return A.first_stage_model.decode(z,force_not_quantize=E or H)
		else:return A.first_stage_model.decode(z)
	def differentiable_decode_first_stage(A,z,predict_cids=_B,force_not_quantize=_B):
		H=force_not_quantize;E=predict_cids
		if E:
			if z.dim()==4:z=torch.argmax(z.exp(),dim=1).long()
			z=A.first_stage_model.quantize.get_codebook_entry(z,shape=_A);z=rearrange(z,_j).contiguous()
		z=_D/A.scale_factor*z
		if hasattr(A,_L):
			if A.split_input_params[_W]:
				B=A.split_input_params[_M];D=A.split_input_params[_N];K=A.split_input_params[_X];P,Q,F,G=z.shape
				if B[0]>F or B[1]>G:B=min(B[0],F),min(B[1],G);print(_Y)
				if D[0]>F or D[1]>G:D=min(D[0],F),min(D[1],G);print(_Z)
				L,M,N,O=A.get_fold_unfold(z,B,D,uf=K);z=M(z);z=z.view((z.shape[0],-1,B[0],B[1],z.shape[-1]))
				if isinstance(A.first_stage_model,VQModelInterface):J=[A.first_stage_model.decode(z[:,:,:,:,B],force_not_quantize=E or H)for B in range(z.shape[-1])]
				else:J=[A.first_stage_model.decode(z[:,:,:,:,B])for B in range(z.shape[-1])]
				C=torch.stack(J,axis=-1);C=C*O;C=C.view((C.shape[0],-1,C.shape[-1]));I=L(C);I=I/N;return I
			elif isinstance(A.first_stage_model,VQModelInterface):return A.first_stage_model.decode(z,force_not_quantize=E or H)
			else:return A.first_stage_model.decode(z)
		elif isinstance(A.first_stage_model,VQModelInterface):return A.first_stage_model.decode(z,force_not_quantize=E or H)
		else:return A.first_stage_model.decode(z)
	@torch.no_grad()
	def encode_first_stage(self,x):
		A=self
		if hasattr(A,_L):
			if A.split_input_params[_W]:
				B=A.split_input_params[_M];D=A.split_input_params[_N];I=A.split_input_params[_X];A.split_input_params[_a]=x.shape[-2:];O,P,F,G=x.shape
				if B[0]>F or B[1]>G:B=min(B[0],F),min(B[1],G);print(_Y)
				if D[0]>F or D[1]>G:D=min(D[0],F),min(D[1],G);print(_Z)
				J,K,L,M=A.get_fold_unfold(x,B,D,df=I);E=K(x);E=E.view((E.shape[0],-1,B[0],B[1],E.shape[-1]));N=[A.first_stage_model.encode(E[:,:,:,:,B])for B in range(E.shape[-1])];C=torch.stack(N,axis=-1);C=C*M;C=C.view((C.shape[0],-1,C.shape[-1]));H=J(C);H=H/L;return H
			else:return A.first_stage_model.encode(x)
		else:return A.first_stage_model.encode(x)
	def shared_step(A,batch,**E):B,C=A.get_input(batch,A.first_stage_key);D=A(B,C);return D
	def forward(A,x,c,*C,**D):
		B=torch.randint(0,A.num_timesteps,(x.shape[0],),device=A.device).long()
		if A.model.conditioning_key is not _A:
			assert c is not _A
			if A.cond_stage_trainable:c=A.get_learned_conditioning(c)
			if A.shorten_cond_schedule:E=A.cond_ids[B].to(A.device);c=A.q_sample(x_start=c,t=E,noise=torch.randn_like(c.float()))
		return A.p_losses(x,c,B,*C,**D)
	def apply_model(A,x_noisy,t,cond,return_ids=_B):
		M=return_ids;H=x_noisy;B=cond
		if isinstance(B,dict):0
		else:
			if not isinstance(B,list):B=[B]
			U=_H if A.model.conditioning_key==_J else _F;B={U:B}
		if hasattr(A,_L):
			assert len(B)==1;assert not M;F=A.split_input_params[_M];I=A.split_input_params[_N];e,V=H.shape[-2:];W,N,X,Y=A.get_fold_unfold(H,F,I);D=N(H);D=D.view((D.shape[0],-1,F[0],F[1],D.shape[-1]));Z=[D[:,:,:,:,A]for A in range(D.shape[-1])]
			if A.cond_stage_key in[_R,'LR_image','segmentation','bbox_img']and A.model.conditioning_key:a=next(iter(B.keys()));C=next(iter(B.values()));assert len(C)==1;C=C[0];C=N(C);C=C.view((C.shape[0],-1,F[0],F[1],C.shape[-1]));L=[{a:[C[:,:,:,:,A]]}for A in range(C.shape[-1])]
			elif A.cond_stage_key==_k:assert _a in A.split_input_params,'BoudingBoxRescaling is missing original_image_size';O=int((V-F[0])/I[0]+1);P,Q=A.split_input_params[_a];b=A.first_stage_model.encoder.num_resolutions-1;J=2**b;c=[(J*I[0]*(A%O)/Q,J*I[1]*(A//O)/P)for A in range(D.shape[-1])];d=[(A,B,J*F[0]/Q,J*F[1]/P)for(A,B)in c];R=[torch.LongTensor(A.bbox_tokenizer._crop_encoder(B))[_A].to(A.device)for B in d];print(R[0].shape);assert isinstance(B,dict),'cond must be dict to be fed into model';S=B[_F][0][...,:-2].to(A.device);print(S.shape);E=torch.stack([torch.cat([S,A],dim=1)for A in R]);E=rearrange(E,'l b n -> (l b) n');print(E.shape);E=A.get_learned_conditioning(E);print(E.shape);E=rearrange(E,'(l b) n d -> l b n d',l=D.shape[-1]);print(E.shape);L=[{_F:[A]}for A in E]
			else:L=[B for A in range(D.shape[-1])]
			T=[A.model(Z[B],t,**L[B])for B in range(D.shape[-1])];assert not isinstance(T[0],tuple);G=torch.stack(T,axis=-1);G=G*Y;G=G.view((G.shape[0],-1,G.shape[-1]));K=W(G)/X
		else:K=A.model(H,t,**B)
		if isinstance(K,tuple)and not M:return K[0]
		else:return K
	def _predict_eps_from_xstart(B,x_t,t,pred_xstart):A=x_t;return(extract_into_tensor(B.sqrt_recip_alphas_cumprod,t,A.shape)*A-pred_xstart)/extract_into_tensor(B.sqrt_recipm1_alphas_cumprod,t,A.shape)
	def _prior_bpd(B,x_start):"\n        Get the prior KL term for the variational lower-bound, measured in\n        bits-per-dim.\n        This term can't be optimized, as it only depends on the encoder.\n        :param x_start: the [N x C x ...] tensor of inputs.\n        :return: a batch of [N] KL values (in bits), one per batch element.\n        ";A=x_start;C=A.shape[0];D=torch.tensor([B.num_timesteps-1]*C,device=A.device);E,H,F=B.q_mean_variance(A,D);G=normal_kl(mean1=E,logvar1=F,mean2=_E,logvar2=_E);return mean_flat(G)/np.log(2.)
	def p_losses(A,x_start,cond,t,noise=_A):
		G=x_start;D=noise;D=default(D,lambda:torch.randn_like(G));L=A.q_sample(x_start=G,t=t,noise=D);I=A.apply_model(L,t,cond);B={};E=_S if A.training else'val'
		if A.parameterization==_I:H=G
		elif A.parameterization==_G:H=D
		else:raise NotImplementedError()
		J=A.get_loss(I,H,mean=_B).mean([1,2,3]);B.update({f"{E}/loss_simple":J.mean()});K=A.logvar[t].to(A.device);C=J/torch.exp(K)+K
		if A.learn_logvar:B.update({f"{E}/loss_gamma":C.mean()});B.update({'logvar':A.logvar.data.mean()})
		C=A.l_simple_weight*C.mean();F=A.get_loss(I,H,mean=_B).mean(dim=(1,2,3));F=(A.lvlb_weights[t]*F).mean();B.update({f"{E}/loss_vlb":F});C+=A.original_elbo_weight*F;B.update({f"{E}/loss":C});return C,B
	def p_mean_variance(A,x,c,t,clip_denoised,return_codebook_ids=_B,quantize_denoised=_B,return_x0=_B,score_corrector=_A,corrector_kwargs=_A):
		H=score_corrector;D=return_codebook_ids;J=t;B=A.apply_model(x,J,c,return_ids=D)
		if H is not _A:assert A.parameterization==_G;B=H.modify_score(A,B,x,t,c,**corrector_kwargs)
		if D:B,K=B
		if A.parameterization==_G:C=A.predict_start_from_noise(x,t=t,noise=B)
		elif A.parameterization==_I:C=B
		else:raise NotImplementedError()
		if clip_denoised:C.clamp_(-_D,_D)
		if quantize_denoised:C,I,[I,I,L]=A.first_stage_model.quantize(C)
		E,F,G=A.q_posterior(x_start=C,x_t=x,t=t)
		if D:return E,F,G,K
		elif return_x0:return E,F,G,C
		else:return E,F,G
	@torch.no_grad()
	def p_sample(self,x,c,t,clip_denoised=_B,repeat_noise=_B,return_codebook_ids=_B,quantize_denoised=_B,return_x0=_B,temperature=_D,noise_dropout=_E,score_corrector=_A,corrector_kwargs=_A):
		I=noise_dropout;E=return_x0;D=return_codebook_ids;J,*F,K=*x.shape,x.device;G=self.p_mean_variance(x=x,c=c,t=t,clip_denoised=clip_denoised,return_codebook_ids=D,quantize_denoised=quantize_denoised,return_x0=E,score_corrector=score_corrector,corrector_kwargs=corrector_kwargs)
		if D:raise DeprecationWarning('Support dropped.');A,F,B,L=G
		elif E:A,F,B,M=G
		else:A,F,B=G
		C=noise_like(x.shape,K,repeat_noise)*temperature
		if I>_E:C=torch.nn.functional.dropout(C,p=I)
		H=(1-(t==0).float()).reshape(J,*(1,)*(len(x.shape)-1))
		if D:return A+H*(.5*B).exp()*C,L.argmax(dim=1)
		if E:return A+H*(.5*B).exp()*C,M
		else:return A+H*(.5*B).exp()*C
	@torch.no_grad()
	def progressive_denoising(self,cond,shape,verbose=_C,callback=_A,quantize_denoised=_B,img_callback=_A,mask=_A,x0=_A,temperature=_D,noise_dropout=_E,score_corrector=_A,corrector_kwargs=_A,batch_size=_A,x_T=_A,start_T=_A,log_every_t=_A):
		N=start_T;M=img_callback;L=callback;J=log_every_t;I=mask;H=temperature;G=shape;C=batch_size;B=self;A=cond
		if not J:J=B.log_every_t
		D=B.num_timesteps
		if C is not _A:O=C if C is not _A else G[0];G=[C]+list(G)
		else:O=C=G[0]
		if x_T is _A:E=torch.randn(G,device=B.device)
		else:E=x_T
		P=[]
		if A is not _A:
			if isinstance(A,dict):A={B:A[B][:C]if not isinstance(A[B],list)else[A[:C]for A in A[B]]for B in A}
			else:A=[A[:C]for A in A]if isinstance(A,list)else A[:C]
		if N is not _A:D=min(D,N)
		Q=tqdm(reversed(range(0,D)),desc=_l,total=D)if verbose else reversed(range(0,D))
		if type(H)==float:H=[H]*D
		for F in Q:
			K=torch.full((O,),F,device=B.device,dtype=torch.long)
			if B.shorten_cond_schedule:assert B.model.conditioning_key!=_O;R=B.cond_ids[K].to(A.device);A=B.q_sample(x_start=A,t=R,noise=torch.randn_like(A))
			E,S=B.p_sample(E,A,K,clip_denoised=B.clip_denoised,quantize_denoised=quantize_denoised,return_x0=_C,temperature=H[F],noise_dropout=noise_dropout,score_corrector=score_corrector,corrector_kwargs=corrector_kwargs)
			if I is not _A:assert x0 is not _A;T=B.q_sample(x0,K);E=T*I+(_D-I)*E
			if F%J==0 or F==D-1:P.append(S)
			if L:L(F)
			if M:M(E,F)
		return E,P
	@torch.no_grad()
	def p_sample_loop(self,cond,shape,return_intermediates=_B,x_T=_A,verbose=_C,callback=_A,timesteps=_A,quantize_denoised=_B,mask=_A,x0=_A,img_callback=_A,start_T=_A,log_every_t=_A):
		L=start_T;K=img_callback;J=callback;I=shape;G=log_every_t;E=mask;D=cond;C=timesteps;A=self
		if not G:G=A.log_every_t
		M=A.betas.device;O=I[0]
		if x_T is _A:B=torch.randn(I,device=M)
		else:B=x_T
		N=[B]
		if C is _A:C=A.num_timesteps
		if L is not _A:C=min(C,L)
		P=tqdm(reversed(range(0,C)),desc=_b,total=C)if verbose else reversed(range(0,C))
		if E is not _A:assert x0 is not _A;assert x0.shape[2:3]==E.shape[2:3]
		for F in P:
			H=torch.full((O,),F,device=M,dtype=torch.long)
			if A.shorten_cond_schedule:assert A.model.conditioning_key!=_O;Q=A.cond_ids[H].to(D.device);D=A.q_sample(x_start=D,t=Q,noise=torch.randn_like(D))
			B=A.p_sample(B,D,H,clip_denoised=A.clip_denoised,quantize_denoised=quantize_denoised)
			if E is not _A:R=A.q_sample(x0,H);B=R*E+(_D-E)*B
			if F%G==0 or F==C-1:N.append(B)
			if J:J(F)
			if K:K(B,F)
		if return_intermediates:return B,N
		return B
	@torch.no_grad()
	def sample(self,cond,batch_size=16,return_intermediates=_B,x_T=_A,verbose=_C,timesteps=_A,quantize_denoised=_B,mask=_A,x0=_A,shape=_A,**E):
		D=shape;C=self;B=batch_size;A=cond
		if D is _A:D=B,C.channels,C.image_size,C.image_size
		if A is not _A:
			if isinstance(A,dict):A={C:A[C][:B]if not isinstance(A[C],list)else[A[:B]for A in A[C]]for C in A}
			else:A=[A[:B]for A in A]if isinstance(A,list)else A[:B]
		return C.p_sample_loop(A,D,return_intermediates=return_intermediates,x_T=x_T,verbose=verbose,timesteps=timesteps,quantize_denoised=quantize_denoised,mask=mask,x0=x0)
	@torch.no_grad()
	def sample_log(self,cond,batch_size,ddim,ddim_steps,**C):
		B=batch_size;A=self
		if ddim:F=DDIMSampler(A);G=A.channels,A.image_size,A.image_size;D,E=F.sample(ddim_steps,B,G,cond,verbose=_B,**C)
		else:D,E=A.sample(cond=cond,batch_size=B,return_intermediates=_C,**C)
		return D,E
	@torch.no_grad()
	def log_images(self,batch,N=4,n_row=4,sample=_C,ddim_steps=200,ddim_eta=_D,return_keys=_A,quantize_denoised=_C,inpaint=_B,plot_denoise_rows=_B,plot_progressive_rows=_B,plot_diffusion_rows=_B,**g):
		Y='caption';T=return_keys;S=batch;R='conditioning';P=ddim_eta;O=ddim_steps;M=n_row;A=self;Q=_B;B={};J,H,D,Z,C=A.get_input(S,A.first_stage_key,return_first_stage_outputs=_C,force_c_encode=_C,return_original_cond=_C,bs=N,uncond=0);N=min(D.shape[0],N);M=min(D.shape[0],M);B[_c]=D;B['reals']=C[_H];B['reconstruction']=Z
		if A.model.conditioning_key is not _A:
			if hasattr(A.cond_stage_model,'decode'):C=A.cond_stage_model.decode(H);B[R]=C
			elif A.cond_stage_key in[Y]:C=log_txt_as_img((D.shape[2],D.shape[3]),S[Y]);B[R]=C
			elif A.cond_stage_key=='class_label':C=log_txt_as_img((D.shape[2],D.shape[3]),S['human_label']);B[R]=C
			elif isimage(C):B[R]=C
			if ismap(C):B['original_conditioning']=A.to_rgb(C)
		if plot_diffusion_rows:
			K=[];W=J[:M]
			for E in range(A.num_timesteps):
				if E%A.log_every_t==0 or E==A.num_timesteps-1:E=repeat(torch.tensor([E]),_d,b=M);E=E.to(A.device).long();a=torch.randn_like(W);b=A.q_sample(x_start=W,t=E,noise=a);K.append(A.decode_first_stage(b))
			K=torch.stack(K);L=rearrange(K,_T);L=rearrange(L,_U);L=make_grid(L,nrow=K.shape[0]);B[_e]=L
		if sample:
			with A.ema_scope(_f):F,X=A.sample_log(cond=H,batch_size=N,ddim=Q,ddim_steps=O,eta=P)
			G=A.decode_first_stage(F);B[_g]=G
			if plot_denoise_rows:c=A._get_denoise_row_from_list(X);B[_h]=c
			if quantize_denoised and not isinstance(A.first_stage_model,AutoencoderKL)and not isinstance(A.first_stage_model,IdentityFirstStage):
				with A.ema_scope('Plotting Quantized Denoised'):F,X=A.sample_log(cond=H,batch_size=N,ddim=Q,ddim_steps=O,eta=P,quantize_denoised=_C)
				G=A.decode_first_stage(F.to(A.device));B['samples_x0_quantized']=G
			if inpaint:
				U,V=J.shape[2],J.shape[3];I=torch.ones(N,U,V).to(A.device);I[:,U//4:3*U//4,V//4:3*V//4]=_E;I=I[:,_A,...]
				with A.ema_scope('Plotting Inpaint'):F,d=A.sample_log(cond=H,batch_size=N,ddim=Q,eta=P,ddim_steps=O,x0=J[:N],mask=I)
				G=A.decode_first_stage(F.to(A.device));B['samples_inpainting']=G;B['mask']=I
				with A.ema_scope('Plotting Outpaint'):F,d=A.sample_log(cond=H,batch_size=N,ddim=Q,eta=P,ddim_steps=O,x0=J[:N],mask=I)
				G=A.decode_first_stage(F.to(A.device));B['samples_outpainting']=G
		if plot_progressive_rows:
			with A.ema_scope('Plotting Progressives'):h,e=A.progressive_denoising(H,shape=(A.channels,A.image_size,A.image_size),batch_size=N)
			f=A._get_denoise_row_from_list(e,desc=_l);B['progressive_row']=f
		if T:
			if np.intersect1d(list(B.keys()),T).shape[0]==0:return B
			else:return{A:B[A]for A in T}
		return B
	def configure_optimizers(A):
		E=A.learning_rate;B=list(A.model.parameters())
		if A.cond_stage_trainable:print(f"{A.__class__.__name__}: Also optimizing conditioner params!");B=B+list(A.cond_stage_model.parameters())
		if A.learn_logvar:print('Diffusion model optimizing logvar');B.append(A.logvar)
		C=torch.optim.AdamW(B,lr=E)
		if A.use_scheduler:assert'target'in A.scheduler_config;D=instantiate_from_config(A.scheduler_config);print('Setting up LambdaLR scheduler...');D=[{'scheduler':LambdaLR(C,lr_lambda=D.schedule),'interval':'step','frequency':1}];return[C],D
		return C
	@torch.no_grad()
	def to_rgb(self,x):
		A=self;x=x.float()
		if not hasattr(A,'colorize'):A.colorize=torch.randn(3,x.shape[1],1,1).to(x)
		x=nn.functional.conv2d(x,weight=A.colorize);x=2.*(x-x.min())/(x.max()-x.min())-_D;return x
class DiffusionWrapper(pl.LightningModule):
	def __init__(A,diff_model_config,conditioning_key):super().__init__();A.diffusion_model=instantiate_from_config(diff_model_config);A.conditioning_key=conditioning_key;assert A.conditioning_key in[_A,_J,_K,_O,_P]
	def forward(A,x,t,c_concat=_A,c_crossattn=_A):
		F=c_concat;D=c_crossattn
		if A.conditioning_key is _A:B=A.diffusion_model(x,t)
		elif A.conditioning_key==_J:E=torch.cat([x]+F,dim=1);B=A.diffusion_model(E,t)
		elif A.conditioning_key==_K:C=torch.cat(D,1);B=A.diffusion_model(x,t,context=C)
		elif A.conditioning_key==_O:E=torch.cat([x]+F,dim=1);C=torch.cat(D,1);B=A.diffusion_model(E,t,context=C)
		elif A.conditioning_key==_P:C=D[0];B=A.diffusion_model(x,t,y=C)
		else:raise NotImplementedError()
		return B
class Layout2ImgDiffusion(LatentDiffusion):
	def __init__(D,cond_stage_key,*B,**C):A=cond_stage_key;assert A==_k,'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"';super().__init__(*B,cond_stage_key=A,**C)
	def log_images(A,batch,N=8,*F,**G):
		C=batch;D=super().log_images(*F,batch=C,N=N,**G);H=_S if A.training else'validation';B=A.trainer.datamodule.datasets[H];I=B.conditional_builders[A.cond_stage_key];E=[];J=lambda catno:B.get_textual_label(B.get_category_id(catno))
		for K in C[A.cond_stage_key][:N]:L=I.plot(K.detach().cpu(),J,(256,256));E.append(L)
		M=torch.stack(E,dim=0);D['bbox_image']=M;return D