_A='act_layer'
import torch
from packaging import version
from modules import devices
from modules.sd_hijack_utils import CondFunc
class TorchHijackForUnet:
	'\n    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;\n    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64\n    '
	def __getattr__(B,item):
		A=item
		if A=='cat':return B.cat
		if hasattr(torch,A):return getattr(torch,A)
		raise AttributeError(f"'{type(B).__name__}' object has no attribute '{A}'")
	def cat(F,tensors,*D,**E):
		A=tensors
		if len(A)==2:
			B,C=A
			if B.shape[-2:]!=C.shape[-2:]:B=torch.nn.functional.interpolate(B,C.shape[-2:],mode='nearest')
			A=B,C
		return torch.cat(A,*D,**E)
th=TorchHijackForUnet()
def apply_model(orig_func,self,x_noisy,t,cond,**C):
	A=cond
	if isinstance(A,dict):
		for B in A.keys():
			if isinstance(A[B],list):A[B]=[A.to(devices.dtype_unet)if isinstance(A,torch.Tensor)else A for A in A[B]]
			else:A[B]=A[B].to(devices.dtype_unet)if isinstance(A[B],torch.Tensor)else A[B]
	with devices.autocast():return orig_func(self,x_noisy.to(devices.dtype_unet),t.to(devices.dtype_unet),A,**C).float()
class GELUHijack(torch.nn.GELU,torch.nn.Module):
	def __init__(A,*B,**C):torch.nn.GELU.__init__(A,*B,**C)
	def forward(A,x):
		if devices.unet_needs_upcast:return torch.nn.GELU.forward(A.float(),x.float()).to(devices.dtype_unet)
		else:return torch.nn.GELU.forward(A,x)
ddpm_edit_hijack=None
def hijack_ddpm_edit():
	global ddpm_edit_hijack
	if not ddpm_edit_hijack:CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage',first_stage_sub,first_stage_cond);CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage',first_stage_sub,first_stage_cond);ddpm_edit_hijack=CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model',apply_model,unet_needs_upcast)
unet_needs_upcast=lambda*A,**B:devices.unet_needs_upcast
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model',apply_model,unet_needs_upcast)
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding',lambda orig_func,timesteps,*A,**B:orig_func(timesteps,*A,**B).to(torch.float32 if timesteps.dtype==torch.int64 else devices.dtype_unet),unet_needs_upcast)
if version.parse(torch.__version__)<=version.parse('1.13.2')or torch.cuda.is_available():CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward',lambda orig_func,self,*A,**B:orig_func(self.float(),*A,**B),unet_needs_upcast);CondFunc('ldm.modules.attention.GEGLU.forward',lambda orig_func,self,x:orig_func(self.float(),x.float()).to(devices.dtype_unet),unet_needs_upcast);CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__',lambda orig_func,*B,**A:A.update({_A:GELUHijack})and False or orig_func(*B,**A),lambda _,*B,**A:A.get(_A)is None or A[_A]==torch.nn.GELU)
first_stage_cond=lambda _,self,*A,**B:devices.unet_needs_upcast and self.model.diffusion_model.dtype==torch.float16
first_stage_sub=lambda orig_func,self,x,**A:orig_func(self,x.to(devices.dtype_vae),**A)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage',first_stage_sub,first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage',first_stage_sub,first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding',lambda orig_func,*A,**B:orig_func(*A,**B).float(),first_stage_cond)
CondFunc('sgm.modules.diffusionmodules.wrappers.OpenAIWrapper.forward',apply_model,unet_needs_upcast)
CondFunc('sgm.modules.diffusionmodules.openaimodel.timestep_embedding',lambda orig_func,timesteps,*A,**B:orig_func(timesteps,*A,**B).to(torch.float32 if timesteps.dtype==torch.int64 else devices.dtype_unet),unet_needs_upcast)