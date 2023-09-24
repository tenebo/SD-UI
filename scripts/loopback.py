_D='DeepBooru'
_C='Aggressive'
_B='Final denoising strength'
_A='None'
import math,gradio as gr,modules.scripts as scripts
from modules import deepbooru,images,processing,shared
from modules.processing import Processed
from modules.shared import opts,state
class Script(scripts.Script):
	def title(A):return'Loopback'
	def show(A,is_img2img):return is_img2img
	def ui(A,is_img2img):B='Linear';C=gr.Slider(minimum=1,maximum=32,step=1,label='Loops',value=4,elem_id=A.elem_id('loops'));D=gr.Slider(minimum=0,maximum=1,step=.01,label=_B,value=.5,elem_id=A.elem_id('final_denoising_strength'));E=gr.Dropdown(label='Denoising strength curve',choices=[_C,B,'Lazy'],value=B);F=gr.Dropdown(label='Append interrogated prompt at each iteration',choices=[_A,'CLIP',_D],value=_A);return[C,D,E,F]
	def run(X,p,loops,final_denoising_strength,denoising_curve,append_interrogation):
		L=final_denoising_strength;K=append_interrogation;J=denoising_curve;F=None;C=loops;processing.fix_seed(p);D=p.n_iter;p.extra_generation_params={_B:L,'Denoising curve':J};p.batch_size=1;p.n_iter=1;R=F;G=F;M=F;H=p.denoising_strength;N=[];E=[];S=p.init_images;O=p.prompt;T=p.inpainting_fill;state.job_count=C*D;U=[processing.setup_color_correction(p.init_images[0])]
		def V(loop):
			A=H
			if C==1:return A
			B=loop/(C-1)
			if J==_C:A=math.sin(B*math.pi*.5)
			elif J=='Lazy':A=1-math.cos(B*math.pi*.5)
			else:A=B
			D=(L-H)*A;return H+D
		I=[]
		for W in range(D):
			p.init_images=S;p.denoising_strength=H;A=F
			for P in range(C):
				p.n_iter=1;p.batch_size=1;p.do_not_save_grid=True
				if opts.img2img_color_correction:p.color_corrections=U
				if K!=_A:
					p.prompt=f"{O}, "if O else''
					if K=='CLIP':p.prompt+=shared.interrogator.interrogate(p.init_images[0])
					elif K==_D:p.prompt+=deepbooru.model.tag(p.init_images[0])
				state.job=f"Iteration {P+1}/{C}, batch {W+1}/{D}";B=processing.process_images(p)
				if state.interrupted:break
				if G is F:G=B.seed;M=B.info
				p.seed=B.seed+1;p.denoising_strength=V(P+1)
				if state.skipped:break
				A=B.images[0];p.init_images=[A];p.inpainting_fill=1
				if D==1:I.append(A);E.append(A)
			if D>1 and not state.skipped and not state.interrupted:I.append(A);E.append(A)
			p.inpainting_fill=T
			if state.interrupted:break
		if len(I)>1:
			Q=images.image_grid(I,rows=1)
			if opts.grid_save:images.save_image(Q,p.outpath_grids,'grid',G,p.prompt,opts.grid_format,info=R,short_filename=not opts.grid_extended_filename,grid=True,p=p)
			if opts.return_grid:N.append(Q)
		E=N+E;B=Processed(p,E,G,M);return B