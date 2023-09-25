import os,torch
from torch import nn
from modules import devices,paths,shared
sd_vae_approx_models={}
class VAEApprox(nn.Module):
	def __init__(A):super(VAEApprox,A).__init__();A.conv1=nn.Conv2d(4,8,(7,7));A.conv2=nn.Conv2d(8,16,(5,5));A.conv3=nn.Conv2d(16,32,(3,3));A.conv4=nn.Conv2d(32,64,(3,3));A.conv5=nn.Conv2d(64,32,(3,3));A.conv6=nn.Conv2d(32,16,(3,3));A.conv7=nn.Conv2d(16,8,(3,3));A.conv8=nn.Conv2d(8,3,(3,3))
	def forward(A,x):
		B=11;x=nn.functional.interpolate(x,(x.shape[2]*2,x.shape[3]*2));x=nn.functional.pad(x,(B,B,B,B))
		for C in[A.conv1,A.conv2,A.conv3,A.conv4,A.conv5,A.conv6,A.conv7,A.conv8]:x=C(x);x=nn.functional.leaky_relu(x,.1)
		return x
def download_model(model_path,model_url):
	A=model_path
	if not os.path.exists(A):os.makedirs(os.path.dirname(A),exist_ok=True);print(f"Downloading VAEApprox model to: {A}");torch.hub.download_url_to_file(model_url,A)
def model():
	D='VAE-approx';B='vaeapprox-sdxl.pt'if getattr(shared.sd_model,'is_sdxl',False)else'model.pt';A=sd_vae_approx_models.get(B)
	if A is None:
		C=os.path.join(paths.models_path,D,B)
		if not os.path.exists(C):C=os.path.join(paths.script_path,'models',D,B)
		if not os.path.exists(C):C=os.path.join(paths.models_path,D,B);download_model(C,'https://github.com/tenebo/standard-demo-ourui/releases/download/v1.0.0-pre/'+B)
		A=VAEApprox();A.load_state_dict(torch.load(C,map_location='cpu'if devices.device.type!='cuda'else None));A.eval();A.to(devices.device,devices.dtype);sd_vae_approx_models[B]=A
	return A
def cheap_approximation(sample):
	A=sample
	if shared.sd_model.is_sdxl:B=[[.3448,.4168,.4395],[-.1953,-.029,.025],[.1074,.0886,-.0163],[-.373,-.2499,-.2088]]
	else:B=[[.298,.207,.208],[.187,.286,.173],[-.158,.189,.264],[-.184,-.271,-.473]]
	C=torch.tensor(B).to(A.device);D=torch.einsum('...lxy,lr -> ...rxy',A,C);return D