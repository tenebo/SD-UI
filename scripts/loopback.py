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
		L=final_denoising_strength;J=append_interrogation;K=denoising_curve;C=None;D=loops;processing.fix_seed(p);E=p.n_iter;p.extra_generation_params={_B:L,'Denoising curve':K};p.batch_size=1;p.n_iter=1;R=C;G=C;M=C;H=p.denoising_strength;N=[];F=[];S=p.init_images;O=p.prompt;T=p.inpainting_fill;state.job_count=D*E;U=[processing.setup_color_correction(p.init_images[0])]
		def V(loop):
			A=H
			if D==1:return A
			B=loop/(D-1)
			if K==_C:A=math.sin(B*math.pi*.5)
			elif K=='Lazy':A=1-math.cos(B*math.pi*.5)
			else:A=B
			C=(L-H)*A;return H+C
		I=[]
		for W in range(E):
			p.init_images=S;p.denoising_strength=H;A=C
			for P in range(D):
				p.n_iter=1;p.batch_size=1;p.do_not_save_grid=True
				if opts.img2img_color_correction:p.color_corrections=U
				if J!=_A:
					p.prompt=f"{O}, "if O else''
					if J=='CLIP':p.prompt+=shared.interrogator.interrogate(p.init_images[0])
					elif J==_D:p.prompt+=deepbooru.model.tag(p.init_images[0])
				state.job=f"Iteration {P+1}/{D}, batch {W+1}/{E}";B=processing.process_images(p)
				if state.interrupted:break
				if G is C:G=B.seed;M=B.info
				p.seed=B.seed+1;p.denoising_strength=V(P+1)
				if state.skipped:break
				A=B.images[0];p.init_images=[A];p.inpainting_fill=1
				if E==1:I.append(A);F.append(A)
			if E>1 and not state.skipped and not state.interrupted:I.append(A);F.append(A)
			p.inpainting_fill=T
			if state.interrupted:break
		if len(I)>1:
			Q=images.image_grid(I,rows=1)
			if opts.grid_save:images.save_image(Q,p.outpath_grids,'grid',G,p.prompt,opts.grid_format,info=R,short_filename=not opts.grid_extended_filename,grid=True,p=p)
			if opts.return_grid:N.append(Q)
		F=N+F;B=Processed(p,F,G,M);return B