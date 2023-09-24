_C='Windows'
_B='SWIN_torch_compile'
_A=None
import sys,platform,numpy as np,torch
from PIL import Image
from tqdm import tqdm
from modules import modelloader,devices,script_callbacks,shared
from modules.shared import opts,state
from swinir_model_arch import SwinIR
from swinir_model_arch_v2 import Swin2SR
from modules.upscaler import Upscaler,UpscalerData
SWINIR_MODEL_URL='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
device_swinir=devices.get_device_for('swinir')
class UpscalerSwinIR(Upscaler):
	def __init__(A,dirname):
		A._cached_model=_A;A._cached_model_config=_A;A.name='SwinIR';A.model_url=SWINIR_MODEL_URL;A.model_name='SwinIR 4x';A.user_path=dirname;super().__init__();C=[];E=A.find_models(ext_filter=['.pt','.pth'])
		for B in E:
			if B.startswith('http'):D=A.model_name
			else:D=modelloader.friendly_name(B)
			F=UpscalerData(D,B,A);C.append(F)
		A.scalers=C
	def do_upscale(B,img,model_file):
		D=model_file;C=img;E=hasattr(opts,_B)and opts.SWIN_torch_compile and int(torch.__version__.split('.')[0])>=2 and platform.system()!=_C;F=D,opts.SWIN_tile
		if E and B._cached_model_config==F:A=B._cached_model
		else:
			B._cached_model=_A
			try:A=B.load_model(D)
			except Exception as G:print(f"Failed loading SwinIR model {D}: {G}",file=sys.stderr);return C
			A=A.to(device_swinir,dtype=devices.dtype)
			if E:A=torch.compile(A);B._cached_model=A;B._cached_model_config=F
		C=upscale(C,A);devices.torch_gc();return C
	def load_model(E,path,scale=4):
		F='nearest+conv';G=scale;B=path
		if B.startswith('http'):C=modelloader.load_file_from_url(url=B,model_dir=E.model_download_path,file_name=f"{E.model_name.replace(' ','_')}.pth")
		else:C=B
		if C.endswith('.v2.pth'):A=Swin2SR(upscale=G,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[6,6,6,6,6,6],embed_dim=180,num_heads=[6,6,6,6,6,6],mlp_ratio=2,upsampler=F,resi_connection='1conv');D=_A
		else:A=SwinIR(upscale=G,in_chans=3,img_size=64,window_size=8,img_range=1.,depths=[6,6,6,6,6,6,6,6,6],embed_dim=240,num_heads=[8,8,8,8,8,8,8,8,8],mlp_ratio=2,upsampler=F,resi_connection='3conv');D='params_ema'
		H=torch.load(C)
		if D is not _A:A.load_state_dict(H[D],strict=True)
		else:A.load_state_dict(H,strict=True)
		return A
def upscale(img,model,tile=_A,tile_overlap=_A,window_size=8,scale=4):
	F=scale;G=tile_overlap;H=tile;C=window_size;A=img;H=H or opts.SWIN_tile;G=G or opts.SWIN_tile_overlap;A=np.array(A);A=A[:,:,::-1];A=np.moveaxis(A,2,0)/255;A=torch.from_numpy(A).float();A=A.unsqueeze(0).to(device_swinir,dtype=devices.dtype)
	with torch.no_grad(),devices.autocast():
		I,I,D,E=A.size();J=(D//C+1)*C-D;K=(E//C+1)*C-E;A=torch.cat([A,torch.flip(A,[2])],2)[:,:,:D+J,:];A=torch.cat([A,torch.flip(A,[3])],3)[:,:,:,:E+K];B=inference(A,model,H,G,C,F);B=B[...,:D*F,:E*F];B=B.data.squeeze().float().cpu().clamp_(0,1).numpy()
		if B.ndim==3:B=np.transpose(B[[2,1,0],:,:],(1,2,0))
		B=(B*255.).round().astype(np.uint8);return Image.fromarray(B,'RGB')
def inference(img,model,tile,tile_overlap,window_size,scale):
	G=img;A=tile;N,O,E,F=G.size();A=min(A,E,F);assert A%window_size==0,'tile size should be a multiple of window_size';B=scale;I=A-tile_overlap;J=list(range(0,E-A,I))+[E-A];K=list(range(0,F-A,I))+[F-A];H=torch.zeros(N,O,E*B,F*B,dtype=devices.dtype,device=device_swinir).type_as(G);L=torch.zeros_like(H,dtype=devices.dtype,device=device_swinir)
	with tqdm(total=len(J)*len(K),desc='SwinIR tiles')as P:
		for C in J:
			if state.interrupted or state.skipped:break
			for D in K:
				if state.interrupted or state.skipped:break
				Q=G[...,C:C+A,D:D+A];M=model(Q);R=torch.ones_like(M);H[...,C*B:(C+A)*B,D*B:(D+A)*B].add_(M);L[...,C*B:(C+A)*B,D*B:(D+A)*B].add_(R);P.update(1)
	S=H.div_(L);return S
def on_ui_settings():
	D='step';E='maximum';F='minimum';A='Upscaling';B='upscaling';import gradio as C;shared.opts.add_option('SWIN_tile',shared.OptionInfo(192,'Tile size for all SwinIR.',C.Slider,{F:16,E:512,D:16},section=(B,A)));shared.opts.add_option('SWIN_tile_overlap',shared.OptionInfo(8,'Tile overlap, in pixels for SwinIR. Low values = visible seam.',C.Slider,{F:0,E:48,D:1},section=(B,A)))
	if int(torch.__version__.split('.')[0])>=2 and platform.system()!=_C:shared.opts.add_option(_B,shared.OptionInfo(False,'Use torch.compile to accelerate SwinIR.',C.Checkbox,{'interactive':True},section=(B,A)).info('Takes longer on first run'))
script_callbacks.on_ui_settings(on_ui_settings)