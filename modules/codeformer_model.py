_A=None
import os,cv2,torch,modules.face_restoration,modules.shared
from modules import shared,devices,modelloader,errors
from modules.paths import models_path
model_dir='Codeformer'
model_path=os.path.join(models_path,model_dir)
model_url='https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
codeformer=_A
def setup_model(dirname):
	A='CodeFormer';B=True;os.makedirs(model_path,exist_ok=B);C=modules.paths.paths.get(A,_A)
	if C is _A:return
	try:
		from torchvision.transforms.functional import normalize as J;from modules.codeformer.codeformer_arch import CodeFormer as E;from basicsr.utils import img2tensor as K,tensor2img as H;from facelib.utils.face_restoration_helper import FaceRestoreHelper as G;from facelib.detection.retinaface import retinaface as D;I=E
		class F(modules.face_restoration.FaceRestoration):
			def name(B):return A
			def __init__(A,dirname):A.net=_A;A.face_helper=_A;A.cmd_dir=dirname
			def create_models(A):
				if A.net is not _A and A.face_helper is not _A:A.net.to(devices.device_codeformer);return A.net,A.face_helper
				E=modelloader.load_models(model_path,model_url,A.cmd_dir,download_name='codeformer-v0.1.0.pth',ext_filter=['.pth'])
				if len(E)!=0:H=E[0]
				else:print('Unable to load codeformer model.');return _A,_A
				C=I(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=['32','64','128','256']).to(devices.device_codeformer);J=torch.load(H)['params_ema'];C.load_state_dict(J);C.eval()
				if hasattr(D,'device'):D.device=devices.device_codeformer
				F=G(1,face_size=512,crop_ratio=(1,1),det_model='retinaface_resnet50',save_ext='png',use_parse=B,device=devices.device_codeformer);A.net=C;A.face_helper=F;return C,F
			def send_model_to(A,device):B=device;A.net.to(B);A.face_helper.face_det.to(B);A.face_helper.face_parse.to(B)
			def restore(A,np_image,w=_A):
				D=np_image;D=D[:,:,::-1];G=D.shape[0:2];A.create_models()
				if A.net is _A or A.face_helper is _A:return D
				A.send_model_to(devices.device_codeformer);A.face_helper.clean_all();A.face_helper.read_image(D);A.face_helper.get_face_landmarks_5(only_center_face=False,resize=640,eye_dist_threshold=5);A.face_helper.align_warp_face()
				for L in A.face_helper.cropped_faces:
					E=K(L/255.,bgr2rgb=B,float32=B);J(E,(.5,.5,.5),(.5,.5,.5),inplace=B);E=E.unsqueeze(0).to(devices.device_codeformer)
					try:
						with torch.no_grad():I=A.net(E,w=w if w is not _A else shared.opts.code_former_weight,adain=B)[0];F=H(I,rgb2bgr=B,min_max=(-1,1))
						del I;devices.torch_gc()
					except Exception:errors.report('Failed inference for CodeFormer',exc_info=B);F=H(E,rgb2bgr=B,min_max=(-1,1))
					F=F.astype('uint8');A.face_helper.add_restored_face(F)
				A.face_helper.get_inverse_affine(_A);C=A.face_helper.paste_faces_to_input_image();C=C[:,:,::-1]
				if G!=C.shape[0:2]:C=cv2.resize(C,(0,0),fx=G[1]/C.shape[1],fy=G[0]/C.shape[0],interpolation=cv2.INTER_LINEAR)
				A.face_helper.clean_all()
				if shared.opts.face_restoration_unload:A.send_model_to(devices.cpu)
				return C
		global codeformer;codeformer=F(dirname);shared.face_restorers.append(codeformer)
	except Exception:errors.report('Error setting up CodeFormer',exc_info=B)