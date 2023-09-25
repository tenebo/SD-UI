_D='darwin'
_C='cuda'
_B=False
_A=None
import sys,contextlib
from functools import lru_cache
import torch
from modules import errors,shared
if sys.platform==_D:from modules import mac_specific
def has_mps():
	if sys.platform!=_D:return _B
	else:return mac_specific.has_mps
def get_cuda_device_string():
	if shared.cmd_opts.device_id is not _A:return f"cuda:{shared.cmd_opts.device_id}"
	return _C
def get_optimal_device_name():
	if torch.cuda.is_available():return get_cuda_device_string()
	if has_mps():return'mps'
	return'cpu'
def get_optimal_device():return torch.device(get_optimal_device_name())
def get_device_for(task):
	if task in shared.cmd_opts.use_cpu:return cpu
	return get_optimal_device()
def torch_gc():
	if torch.cuda.is_available():
		with torch.cuda.device(get_cuda_device_string()):torch.cuda.empty_cache();torch.cuda.ipc_collect()
	if has_mps():mac_specific.torch_mps_gc()
def enable_tf32():
	A=True
	if torch.cuda.is_available():
		if any(torch.cuda.get_device_capability(A)==(7,5)for A in range(0,torch.cuda.device_count())):torch.backends.cudnn.benchmark=A
		torch.backends.cuda.matmul.allow_tf32=A;torch.backends.cudnn.allow_tf32=A
errors.run(enable_tf32,'Enabling TF32')
cpu=torch.device('cpu')
device=_A
device_interrogate=_A
device_gfpgan=_A
device_esrgan=_A
device_codeformer=_A
dtype=torch.float16
dtype_vae=torch.float16
dtype_unet=torch.float16
unet_needs_upcast=_B
def cond_cast_unet(input):return input.to(dtype_unet)if unet_needs_upcast else input
def cond_cast_float(input):return input.float()if unet_needs_upcast else input
nv_rng=_A
def autocast(disable=_B):
	if disable:return contextlib.nullcontext()
	if dtype==torch.float32 or shared.cmd_opts.precision=='full':return contextlib.nullcontext()
	return torch.autocast(_C)
def without_autocast(disable=_B):return torch.autocast(_C,enabled=_B)if torch.is_autocast_enabled()and not disable else contextlib.nullcontext()
class NansException(Exception):0
def test_for_nans(x,where):
	B=where
	if shared.cmd_opts.disable_nan_check:return
	if not torch.all(torch.isnan(x)).item():return
	if B=='unet':
		A='A tensor with all NaNs was produced in Unet.'
		if not shared.cmd_opts.no_half:A+=' This could be either because there\'s not enough precision to represent the picture, or because your video card does not support half type. Try setting the "Upcast cross attention layer to float32" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this.'
	elif B=='vae':
		A='A tensor with all NaNs was produced in VAE.'
		if not shared.cmd_opts.no_half and not shared.cmd_opts.no_half_vae:A+=" This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
	else:A='A tensor with all NaNs was produced.'
	A+=' Use --disable-nan-check commandline argument to disable this check.';raise NansException(A)
@lru_cache
def first_time_calculation():'\n    just do any calculation with pytorch layers - the first time this is done it allocaltes about 700MB of memory and\n    spends about 2.7 seconds doing that, at least wih NVidia.\n    ';A=torch.zeros((1,1)).to(device,dtype);B=torch.nn.Linear(1,1).to(device,dtype);B(A);A=torch.zeros((1,1,3,3)).to(device,dtype);C=torch.nn.Conv2d(1,1,(3,3)).to(device,dtype);C(A)