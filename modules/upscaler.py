_D='Resampling'
_C=False
_B=True
_A=None
import os
from abc import abstractmethod
import PIL
from PIL import Image
import modules.shared
from modules import modelloader,shared
LANCZOS=Image.Resampling.LANCZOS if hasattr(Image,_D)else Image.LANCZOS
NEAREST=Image.Resampling.NEAREST if hasattr(Image,_D)else Image.NEAREST
class Upscaler:
	name=_A;model_path=_A;model_name=_A;model_url=_A;enable=_B;filter=_A;model=_A;user_path=_A;scalers:0;tile=_B
	def __init__(A,create_dirs=_C):
		A.mod_pad_h=_A;A.tile_size=modules.shared.opts.ESRGAN_tile;A.tile_pad=modules.shared.opts.ESRGAN_tile_overlap;A.device=modules.shared.device;A.img=_A;A.output=_A;A.scale=1;A.half=not modules.shared.cmd_opts.no_half;A.pre_pad=0;A.mod_scale=_A;A.model_download_path=_A
		if A.model_path is _A and A.name:A.model_path=os.path.join(shared.models_path,A.name)
		if A.model_path and create_dirs:os.makedirs(A.model_path,exist_ok=_B)
		try:import cv2;A.can_tile=_B
		except Exception:pass
	@abstractmethod
	def do_upscale(self,img,selected_model):return img
	def upscale(E,img,scale,selected_model=_A):
		B=scale;A=img;E.scale=B;C=int(A.width*B//8*8);D=int(A.height*B//8*8)
		for G in range(3):
			F=A.width,A.height;A=E.do_upscale(A,selected_model)
			if F==(A.width,A.height):break
			if A.width>=C and A.height>=D:break
		if A.width!=C or A.height!=D:A=A.resize((int(C),int(D)),resample=LANCZOS)
		return A
	@abstractmethod
	def load_model(self,path):0
	def find_models(A,ext_filter=_A):return modelloader.load_models(model_path=A.model_path,model_url=A.model_url,command_path=A.user_path,ext_filter=ext_filter)
	def update_status(A,prompt):print(f"\nextras: {prompt}",file=shared.progress_print_out)
class UpscalerData:
	name=_A;data_path=_A;scale=4;scaler=_A;model:0
	def __init__(A,name,path,upscaler=_A,scale=4,model=_A):A.name=name;A.data_path=path;A.local_data_path=path;A.scaler=upscaler;A.scale=scale;A.model=model
class UpscalerNone(Upscaler):
	name='None';scalers=[]
	def load_model(A,path):0
	def do_upscale(A,img,selected_model=_A):return img
	def __init__(A,dirname=_A):super().__init__(_C);A.scalers=[UpscalerData('None',_A,A)]
class UpscalerLanczos(Upscaler):
	scalers=[]
	def do_upscale(B,img,selected_model=_A):A=img;return A.resize((int(A.width*B.scale),int(A.height*B.scale)),resample=LANCZOS)
	def load_model(A,_):0
	def __init__(A,dirname=_A):B='Lanczos';super().__init__(_C);A.name=B;A.scalers=[UpscalerData(B,_A,A)]
class UpscalerNearest(Upscaler):
	scalers=[]
	def do_upscale(B,img,selected_model=_A):A=img;return A.resize((int(A.width*B.scale),int(A.height*B.scale)),resample=NEAREST)
	def load_model(A,_):0
	def __init__(A,dirname=_A):B='Nearest';super().__init__(_C);A.name=B;A.scalers=[UpscalerData(B,_A,A)]