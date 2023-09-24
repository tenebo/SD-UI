_A=None
import torch.nn,ldm.modules.diffusionmodules.openaimodel
from modules import script_callbacks,shared,devices
unet_options=[]
current_unet_option=_A
current_unet=_A
def list_unets():A=script_callbacks.list_unets_callback();unet_options.clear();unet_options.extend(A)
def get_unet_option(option=_A):
	C='None';A=option;A=A or shared.opts.sd_unet
	if A==C:return
	if A=='Automatic':D=shared.sd_model.sd_checkpoint_info.model_name;B=[A for A in unet_options if A.model_name==D];A=B[0].label if B else C
	return next(iter([B for B in unet_options if B.label==A]),_A)
def apply_unet(option=_A):
	global current_unet_option;global current_unet;A=get_unet_option(option)
	if A==current_unet_option:return
	if current_unet is not _A:print(f"Dectivating unet: {current_unet.option.label}");current_unet.deactivate()
	current_unet_option=A
	if current_unet_option is _A:
		current_unet=_A
		if not shared.sd_model.lowvram:shared.sd_model.model.diffusion_model.to(devices.device)
		return
	shared.sd_model.model.diffusion_model.to(devices.cpu);devices.torch_gc();current_unet=current_unet_option.create_unet();current_unet.option=current_unet_option;print(f"Activating unet: {current_unet.option.label}");current_unet.activate()
class SdUnetOption:
	model_name=_A;'name of related checkpoint - this option will be selected automatically for unet if the name of checkpoint matches this';label=_A;'name of the unet in UI'
	def create_unet(A):'returns SdUnet object to be used as a Unet instead of built-in unet when making pictures';raise NotImplementedError()
class SdUnet(torch.nn.Module):
	def forward(A,x,timesteps,context,*B,**C):raise NotImplementedError()
	def activate(A):0
	def deactivate(A):0
def UNetModel_forward(self,x,timesteps=_A,context=_A,*C,**D):
	B=context;A=timesteps
	if current_unet is not _A:return current_unet.forward(x,A,B,*C,**D)
	return ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui(self,x,A,B,*C,**D)