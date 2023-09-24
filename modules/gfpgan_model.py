_D=False
_C='GFPGAN'
_B=True
_A=None
import os,facexlib,gfpgan,modules.face_restoration
from modules import paths,shared,devices,modelloader,errors
model_dir=_C
user_path=_A
model_path=os.path.join(paths.models_path,model_dir)
model_url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
have_gfpgan=_D
loaded_gfpgan_model=_A
def gfpgann():
	global loaded_gfpgan_model;global model_path
	if loaded_gfpgan_model is not _A:loaded_gfpgan_model.gfpgan.to(devices.device_gfpgan);return loaded_gfpgan_model
	if gfpgan_constructor is _A:return
	A=modelloader.load_models(model_path,model_url,user_path,ext_filter=_C)
	if len(A)==1 and A[0].startswith('http'):B=A[0]
	elif len(A)!=0:D=max(A,key=os.path.getctime);B=D
	else:print('Unable to load gfpgan model!');return
	if hasattr(facexlib.detection.retinaface,'device'):facexlib.detection.retinaface.device=devices.device_gfpgan
	C=gfpgan_constructor(model_path=B,upscale=1,arch='clean',channel_multiplier=2,bg_upsampler=_A,device=devices.device_gfpgan);loaded_gfpgan_model=C;return C
def send_model_to(model,device):A=device;B=model;B.gfpgan.to(A);B.face_helper.face_det.to(A);B.face_helper.face_parse.to(A)
def gfpgan_fix_faces(np_image):
	B=np_image;A=gfpgann()
	if A is _A:return B
	send_model_to(A,devices.device_gfpgan);C=B[:,:,::-1];E,F,D=A.enhance(C,has_aligned=_D,only_center_face=_D,paste_back=_B);B=D[:,:,::-1];A.face_helper.clean_all()
	if shared.opts.face_restoration_unload:send_model_to(A,devices.cpu)
	return B
gfpgan_constructor=_A
def setup_model(dirname):
	try:
		os.makedirs(model_path,exist_ok=_B);from gfpgan import GFPGANer as A;from facexlib import detection,parsing;global user_path;global have_gfpgan;global gfpgan_constructor;B=gfpgan.utils.load_file_from_url;C=facexlib.detection.load_file_from_url;D=facexlib.parsing.load_file_from_url
		def E(**A):return B(**dict(A,model_dir=model_path))
		def F(**A):return C(**dict(A,save_dir=model_path,model_dir=_A))
		def G(**A):return D(**dict(A,save_dir=model_path,model_dir=_A))
		gfpgan.utils.load_file_from_url=E;facexlib.detection.load_file_from_url=F;facexlib.parsing.load_file_from_url=G;user_path=dirname;have_gfpgan=_B;gfpgan_constructor=A
		class H(modules.face_restoration.FaceRestoration):
			def name(A):return _C
			def restore(A,np_image):return gfpgan_fix_faces(np_image)
		shared.face_restorers.append(H())
	except Exception:errors.report('Error setting up GFPGAN',exc_info=_B)