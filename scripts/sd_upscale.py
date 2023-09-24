import math,modules.scripts as scripts,gradio as gr
from PIL import Image
from modules import processing,shared,images,devices
from modules.processing import Processed
from modules.shared import opts,state
class Script(scripts.Script):
	def title(A):return'SD upscale'
	def show(A,is_img2img):return is_img2img
	def ui(A,is_img2img):B=gr.HTML('<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>');C=gr.Slider(minimum=0,maximum=256,step=16,label='Tile overlap',value=64,elem_id=A.elem_id('overlap'));D=gr.Slider(minimum=1.,maximum=4.,step=.05,label='Scale Factor',value=2.,elem_id=A.elem_id('scale_factor'));E=gr.Radio(label='Upscaler',choices=[A.name for A in shared.sd_upscalers],value=shared.sd_upscalers[0].name,type='index',elem_id=A.elem_id('upscaler_index'));return[B,C,E,D]
	def run(Y,p,_,overlap,upscaler_index,scale_factor):
		O=overlap;C=upscaler_index
		if isinstance(C,str):C=[A.name.lower()for A in shared.sd_upscalers].index(C.lower())
		processing.fix_seed(p);D=shared.sd_upscalers[C];p.extra_generation_params['SD upscale overlap']=O;p.extra_generation_params['SD upscale upscaler']=D.name;E=None;P=p.seed;F=p.init_images[0];F=images.flatten(F,opts.img2img_background_color)
		if D.name!='None':Q=D.scaler.upscale(F,scale_factor,D.data_path)
		else:Q=F
		devices.torch_gc();A=images.split_grid(Q,tile_w=p.width,tile_h=p.height,overlap=O);G=p.batch_size;R=p.n_iter;p.n_iter=1;p.do_not_save_grid=True;p.do_not_save_samples=True;H=[]
		for(W,X,I)in A.tiles:
			for J in I:H.append(J[2])
		K=math.ceil(len(H)/G);state.job_count=K*R;print(f"SD upscaling will process a total of {len(H)} images tiled as {len(A.tiles[0][2])}x{len(A.tiles)} per upscale in a total of {state.job_count} batches.");S=[]
		for T in range(R):
			U=P+T;p.seed=U;L=[]
			for M in range(K):
				p.batch_size=G;p.init_images=H[M*G:(M+1)*G];state.job=f"Batch {M+1+T*K} out of {state.job_count}";B=processing.process_images(p)
				if E is None:E=B.info
				p.seed=B.seed+1;L+=B.images
			N=0
			for(W,X,I)in A.tiles:
				for J in I:J[2]=L[N]if N<len(L)else Image.new('RGB',(p.width,p.height));N+=1
			V=images.combine_grid(A);S.append(V)
			if opts.samples_save:images.save_image(V,p.outpath_samples,'',U,p.prompt,opts.samples_format,info=E,p=p)
		B=Processed(p,S,P,E);return B