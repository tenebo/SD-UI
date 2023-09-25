_A=True
import os,numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from modules.upscaler import Upscaler,UpscalerData
from modules.shared import cmd_opts,opts
from modules import modelloader,errors
class UpscalerRealESRGAN(Upscaler):
	def __init__(A,path):
		A.name='RealESRGAN';A.user_path=path;super().__init__()
		try:
			from basicsr.archs.rrdbnet_arch import RRDBNet;from realesrgan import RealESRGANer;from realesrgan.archs.srvgg_arch import SRVGGNetCompact;A.enable=_A;A.scalers=[];D=A.load_models(path);E=A.find_models(ext_filter=['.pth'])
			for B in D:
				if B.local_data_path.startswith('http'):
					F=modelloader.friendly_name(B.local_data_path);C=[A for A in E if A.endswith(f"{F}.pth")]
					if C:B.local_data_path=C[0]
				if B.name in opts.realesrgan_enabled_models:A.scalers.append(B)
		except Exception:errors.report('Error importing Real-ESRGAN',exc_info=_A);A.enable=False;A.scalers=[]
	def do_upscale(B,img,path):
		C=img
		if not B.enable:return C
		try:A=B.load_model(path)
		except Exception:errors.report(f"Unable to load RealESRGAN model {path}",exc_info=_A);return C
		D=RealESRGANer(scale=A.scale,model_path=A.local_data_path,model=A.model(),half=not cmd_opts.no_half and not cmd_opts.upcast_sampling,tile=opts.ESRGAN_tile,tile_pad=opts.ESRGAN_tile_overlap,device=B.device);E=D.enhance(np.array(C),outscale=A.scale)[0];F=Image.fromarray(E);return F
	def load_model(B,path):
		for A in B.scalers:
			if A.data_path==path:
				if A.local_data_path.startswith('http'):A.local_data_path=modelloader.load_file_from_url(A.data_path,model_dir=B.model_download_path)
				if not os.path.exists(A.local_data_path):raise FileNotFoundError(f"RealESRGAN data missing: {A.local_data_path}")
				return A
		raise ValueError(f"Unable to find model info: {path}")
	def load_models(A,_):return get_realesrgan_models(A)
def get_realesrgan_models(scaler):
	D='prelu';A=scaler
	try:from basicsr.archs.rrdbnet_arch import RRDBNet as B;from realesrgan.archs.srvgg_arch import SRVGGNetCompact as C;E=[UpscalerData(name='R-ESRGAN General 4xV3',path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',scale=4,upscaler=A,model=lambda:C(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=32,upscale=4,act_type=D)),UpscalerData(name='R-ESRGAN General WDN 4xV3',path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',scale=4,upscaler=A,model=lambda:C(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=32,upscale=4,act_type=D)),UpscalerData(name='R-ESRGAN AnimeVideo',path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth',scale=4,upscaler=A,model=lambda:C(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=16,upscale=4,act_type=D)),UpscalerData(name='R-ESRGAN 4x+',path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',scale=4,upscaler=A,model=lambda:B(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)),UpscalerData(name='R-ESRGAN 4x+ Anime6B',path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',scale=4,upscaler=A,model=lambda:B(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=6,num_grow_ch=32,scale=4)),UpscalerData(name='R-ESRGAN 2x+',path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',scale=2,upscaler=A,model=lambda:B(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=2))];return E
	except Exception:errors.report('Error making Real-ESRGAN models list',exc_info=_A)