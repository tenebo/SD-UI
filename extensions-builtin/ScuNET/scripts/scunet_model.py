_A='scunet'
import sys,PIL.Image,numpy as np,torch
from tqdm import tqdm
import modules.upscaler
from modules import devices,modelloader,script_callbacks,errors
from scunet_model_arch import SCUNet
from modules.modelloader import load_file_from_url
from modules.shared import opts
class UpscalerScuNET(modules.upscaler.Upscaler):
	def __init__(A,dirname):
		A.name='ScuNET';A.model_name='ScuNET GAN';A.model_name2='ScuNET PSNR';A.model_url='https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth';A.model_url2='https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth';A.user_path=dirname;super().__init__();F=A.find_models(ext_filter=['.pth']);C=[];E=True
		for B in F:
			if B.startswith('http'):D=A.model_name
			else:D=modelloader.friendly_name(B)
			if D==A.model_name2 or B==A.model_url2:E=False
			try:G=modules.upscaler.UpscalerData(D,B,A,4);C.append(G)
			except Exception:errors.report(f"Error loading ScuNET model: {B}",exc_info=True)
		if E:H=modules.upscaler.UpscalerData(A.model_name2,A.model_url2,A);C.append(H)
		A.scalers=C
	@staticmethod
	@torch.no_grad()
	def tiled_inference(img,model):
		I=model;E=img;F,G=E.shape[2:];A=opts.SCUNET_tile;P=opts.SCUNET_tile_overlap
		if A==0:return I(E)
		J=devices.get_device_for(_A);assert A%8==0,'tile size should be a multiple of window_size';B=1;K=A-P;L=list(range(0,F-A,K))+[F-A];M=list(range(0,G-A,K))+[G-A];H=torch.zeros(1,3,F*B,G*B,dtype=E.dtype,device=J);N=torch.zeros_like(H,dtype=devices.dtype,device=J)
		with tqdm(total=len(L)*len(M),desc='ScuNET tiles')as Q:
			for C in L:
				for D in M:R=E[...,C:C+A,D:D+A];O=I(R);S=torch.ones_like(O);H[...,C*B:(C+A)*B,D*B:(D+A)*B].add_(O);N[...,C*B:(C+A)*B,D*B:(D+A)*B].add_(S);Q.update(1)
		T=H.div_(N);return T
	def do_upscale(I,img,selected_file):
		J=selected_file;C=img;devices.torch_gc()
		try:L=I.load_model(J)
		except Exception as M:print(f"ScuNET: Unable to load model from {J}: {M}",file=sys.stderr);return C
		N=devices.get_device_for(_A);D=opts.SCUNET_tile;E,F=C.height,C.width;B=np.array(C);B=B[:,:,::-1];B=B.transpose((2,0,1))/255;A=torch.from_numpy(B).float().unsqueeze(0).to(N)
		if D>E or D>F:K=torch.zeros(1,3,max(E,D),max(F,D),dtype=A.dtype,device=A.device);K[:,:,:E,:F]=A;A=K
		G=I.tiled_inference(A,L).squeeze(0);G=G[:,:E*1,:F*1];O=G.float().cpu().clamp_(0,1).numpy();del A,G;devices.torch_gc();H=O.transpose((1,2,0));H=H[:,:,::-1];return PIL.Image.fromarray((H*255).astype(np.uint8))
	def load_model(B,path):
		D=devices.get_device_for(_A)
		if path.startswith('http'):C=load_file_from_url(B.model_url,model_dir=B.model_download_path,file_name=f"{B.name}.pth")
		else:C=path
		A=SCUNet(in_nc=3,config=[4,4,4,4,4,4,4],dim=64);A.load_state_dict(torch.load(C),strict=True);A.eval()
		for(F,E)in A.named_parameters():E.requires_grad=False
		A=A.to(D);return A
def on_ui_settings():B='Upscaling';C='upscaling';D='step';E='maximum';F='minimum';import gradio as G;from modules import shared as A;A.opts.add_option('SCUNET_tile',A.OptionInfo(256,'Tile size for SCUNET upscalers.',G.Slider,{F:0,E:512,D:16},section=(C,B)).info('0 = no tiling'));A.opts.add_option('SCUNET_tile_overlap',A.OptionInfo(8,'Tile overlap for SCUNET upscalers.',G.Slider,{F:0,E:64,D:1},section=(C,B)).info('Low values = visible seam'))
script_callbacks.on_ui_settings(on_ui_settings)