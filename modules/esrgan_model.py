_P='body.0.rdb1.conv1.weight'
_O='conv_last.bias'
_N='conv_last.weight'
_M='model.6.bias'
_L='model.6.weight'
_K='model.3.bias'
_J='model.3.weight'
_I='.0.bias'
_H='.0.weight'
_G='model.1.sub.'
_F='conv_first.bias'
_E='model.0.bias'
_D='.bias'
_C='.weight'
_B='model.0.weight'
_A='conv_first.weight'
import sys,numpy as np,torch
from PIL import Image
import modules.esrgan_model_arch as arch
from modules import modelloader,images,devices
from modules.shared import opts
from modules.upscaler import Upscaler,UpscalerData
def mod2normal(state_dict):
	A=state_dict
	if _A in A:
		B={};E=list(A);B[_B]=A[_A];B[_E]=A[_F]
		for C in E.copy():
			if'RDB'in C:
				D=C.replace('RRDB_trunk.',_G)
				if _C in C:D=D.replace(_C,_H)
				elif _D in C:D=D.replace(_D,_I)
				B[D]=A[C];E.remove(C)
		B['model.1.sub.23.weight']=A['trunk_conv.weight'];B['model.1.sub.23.bias']=A['trunk_conv.bias'];B[_J]=A['upconv1.weight'];B[_K]=A['upconv1.bias'];B[_L]=A['upconv2.weight'];B[_M]=A['upconv2.bias'];B['model.8.weight']=A['HRconv.weight'];B['model.8.bias']=A['HRconv.bias'];B['model.10.weight']=A[_N];B['model.10.bias']=A[_O];A=B
	return A
def resrgan2normal(state_dict,nb=23):
	G='conv_up3.weight';A=state_dict
	if _A in A and _P in A:
		E=0;B={};F=list(A);B[_B]=A[_A];B[_E]=A[_F]
		for D in F.copy():
			if'rdb'in D:
				C=D.replace('body.',_G);C=C.replace('.rdb','.RDB')
				if _C in D:C=C.replace(_C,_H)
				elif _D in D:C=C.replace(_D,_I)
				B[C]=A[D];F.remove(D)
		B[f"model.1.sub.{nb}.weight"]=A['conv_body.weight'];B[f"model.1.sub.{nb}.bias"]=A['conv_body.bias'];B[_J]=A['conv_up1.weight'];B[_K]=A['conv_up1.bias'];B[_L]=A['conv_up2.weight'];B[_M]=A['conv_up2.bias']
		if G in A:E=3;B['model.9.weight']=A[G];B['model.9.bias']=A['conv_up3.bias']
		B[f"model.{8+E}.weight"]=A['conv_hr.weight'];B[f"model.{8+E}.bias"]=A['conv_hr.bias'];B[f"model.{10+E}.weight"]=A[_N];B[f"model.{10+E}.bias"]=A[_O];A=B
	return A
def infer_params(state_dict):
	B=state_dict;G=0;J=6;H=0;C=False
	for D in list(B):
		A=D.split('.');I=len(A)
		if I==5 and A[2]=='sub':K=int(A[3])
		elif I==3:
			E=int(A[1])
			if E>J and A[0]=='model'and A[2]=='weight':G+=1
			if E>H:H=E;F=B[D].shape[0]
		if not C and'conv1x1'in D:C=True
	L=B[_B].shape[0];M=B[_B].shape[1];F=F;N=2**G;return M,F,L,K,C,N
class UpscalerESRGAN(Upscaler):
	def __init__(A,dirname):
		A.name='ESRGAN';A.model_url='https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth';A.model_name='ESRGAN_4x';A.scalers=[];A.user_path=dirname;super().__init__();D=A.find_models(ext_filter=['.pt','.pth']);F=[]
		if len(D)==0:B=UpscalerData(A.model_name,A.model_url,A,4);F.append(B)
		for C in D:
			if C.startswith('http'):E=A.model_name
			else:E=modelloader.friendly_name(C)
			B=UpscalerData(E,C,A,4);A.scalers.append(B)
	def do_upscale(D,img,selected_model):
		B=selected_model;A=img
		try:C=D.load_model(B)
		except Exception as E:print(f"Unable to load ESRGAN model {B}: {E}",file=sys.stderr);return A
		C.to(devices.device_esrgan);A=esrgan_upscale(C,A);return A
	def load_model(D,path):
		G='params';F='params_ema'
		if path.startswith('http'):C=modelloader.load_file_from_url(url=D.model_url,model_dir=D.model_download_path,file_name=f"{D.model_name}.pth")
		else:C=path
		A=torch.load(C,map_location='cpu'if devices.device_esrgan.type=='mps'else None)
		if F in A:A=A[F]
		elif G in A:A=A[G];H=16 if'realesr-animevideov3'in C else 32;B=arch.SRVGGNetCompact(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=H,upscale=4,act_type='prelu');B.load_state_dict(A);B.eval();return B
		if _P in A and _A in A:E=6 if'RealESRGAN_x4plus_anime_6B'in C else 23;A=resrgan2normal(A,E)
		elif _A in A:A=mod2normal(A)
		elif _B not in A:raise Exception('The file is not a recognized ESRGAN model.')
		I,J,K,E,L,M=infer_params(A);B=arch.RRDBNet(in_nc=I,out_nc=J,nf=K,nb=E,upscale=M,plus=L);B.load_state_dict(A);B.eval();return B
def upscale_without_tiling(model,img):
	A=img;A=np.array(A);A=A[:,:,::-1];A=np.ascontiguousarray(np.transpose(A,(2,0,1)))/255;A=torch.from_numpy(A).float();A=A.unsqueeze(0).to(devices.device_esrgan)
	with torch.no_grad():B=model(A)
	B=B.squeeze().float().cpu().clamp_(0,1).numpy();B=255.*np.moveaxis(B,0,2);B=B.astype(np.uint8);B=B[:,:,::-1];return Image.fromarray(B,'RGB')
def esrgan_upscale(model,img):
	D=model
	if opts.ESRGAN_tile==0:return upscale_without_tiling(D,img)
	B=images.split_grid(img,opts.ESRGAN_tile,opts.ESRGAN_tile,opts.ESRGAN_tile_overlap);E=[];A=1
	for(H,I,J)in B.tiles:
		F=[]
		for K in J:L,M,G=K;C=upscale_without_tiling(D,G);A=C.width//G.width;F.append([L*A,M*A,C])
		E.append([H*A,I*A,F])
	N=images.Grid(E,B.tile_w*A,B.tile_h*A,B.image_w*A,B.image_h*A,B.overlap*A);C=images.combine_grid(N);return C