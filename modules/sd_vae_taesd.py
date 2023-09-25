"\nTiny AutoEncoder for Stable Diffusion\n(DNN for encoding / decoding SD's latent space)\n\nhttps://github.com/madebyollin/taesd\n"
_H='TAESD model not found'
_G='https://github.com/madebyollin/taesd/raw/main/'
_F='VAE-taesd'
_E='is_sdxl'
_D='taesd_encoder.pth'
_C='taesd_decoder.pth'
_B=None
_A=False
import os,torch,torch.nn as nn
from modules import devices,paths_internal,shared
sd_vae_taesd_models={}
def conv(n_in,n_out,**A):return nn.Conv2d(n_in,n_out,3,padding=1,**A)
class Clamp(nn.Module):
	@staticmethod
	def forward(x):return torch.tanh(x/3)*3
class Block(nn.Module):
	def __init__(B,n_in,n_out):C=n_in;A=n_out;super().__init__();B.conv=nn.Sequential(conv(C,A),nn.ReLU(),conv(A,A),nn.ReLU(),conv(A,A));B.skip=nn.Conv2d(C,A,1,bias=_A)if C!=A else nn.Identity();B.fuse=nn.ReLU()
	def forward(A,x):return A.fuse(A.conv(x)+A.skip(x))
def decoder():return nn.Sequential(Clamp(),conv(4,64),nn.ReLU(),Block(64,64),Block(64,64),Block(64,64),nn.Upsample(scale_factor=2),conv(64,64,bias=_A),Block(64,64),Block(64,64),Block(64,64),nn.Upsample(scale_factor=2),conv(64,64,bias=_A),Block(64,64),Block(64,64),Block(64,64),nn.Upsample(scale_factor=2),conv(64,64,bias=_A),Block(64,64),conv(64,3))
def encoder():return nn.Sequential(conv(3,64),Block(64,64),conv(64,64,stride=2,bias=_A),Block(64,64),Block(64,64),Block(64,64),conv(64,64,stride=2,bias=_A),Block(64,64),Block(64,64),Block(64,64),conv(64,64,stride=2,bias=_A),Block(64,64),Block(64,64),Block(64,64),conv(64,4))
class TAESDDecoder(nn.Module):
	latent_magnitude=3;latent_shift=.5
	def __init__(A,decoder_path=_C):'Initialize pretrained TAESD on the given device from the given checkpoints.';super().__init__();A.decoder=decoder();A.decoder.load_state_dict(torch.load(decoder_path,map_location='cpu'if devices.device.type!='cuda'else _B))
class TAESDEncoder(nn.Module):
	latent_magnitude=3;latent_shift=.5
	def __init__(A,encoder_path=_D):'Initialize pretrained TAESD on the given device from the given checkpoints.';super().__init__();A.encoder=encoder();A.encoder.load_state_dict(torch.load(encoder_path,map_location='cpu'if devices.device.type!='cuda'else _B))
def download_model(model_path,model_url):
	A=model_path
	if not os.path.exists(A):os.makedirs(os.path.dirname(A),exist_ok=True);print(f"Downloading TAESD model to: {A}");torch.hub.download_url_to_file(model_url,A)
def decoder_model():
	B='taesdxl_decoder.pth'if getattr(shared.sd_model,_E,_A)else _C;A=sd_vae_taesd_models.get(B)
	if A is _B:
		C=os.path.join(paths_internal.models_path,_F,B);download_model(C,_G+B)
		if os.path.exists(C):A=TAESDDecoder(C);A.eval();A.to(devices.device,devices.dtype);sd_vae_taesd_models[B]=A
		else:raise FileNotFoundError(_H)
	return A.decoder
def encoder_model():
	B='taesdxl_encoder.pth'if getattr(shared.sd_model,_E,_A)else _D;A=sd_vae_taesd_models.get(B)
	if A is _B:
		C=os.path.join(paths_internal.models_path,_F,B);download_model(C,_G+B)
		if os.path.exists(C):A=TAESDEncoder(C);A.eval();A.to(devices.device,devices.dtype);sd_vae_taesd_models[B]=A
		else:raise FileNotFoundError(_H)
	return A.encoder