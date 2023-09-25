_C='negative'
_B='comma'
_A='positive'
import math,modules.scripts as scripts,gradio as gr
from modules import images
from modules.processing import process_images
from modules.shared import opts,state
import modules.sd_samplers
def draw_xy_grid(xs,ys,x_label,y_label,cell):
	B=ys;A=xs;C=[];G=[[images.GridAnnotation(y_label(A))]for A in B];H=[[images.GridAnnotation(x_label(A))]for A in A];D=None;state.job_count=len(A)*len(B)
	for(I,J)in enumerate(B):
		for(K,L)in enumerate(A):
			state.job=f"{K+I*len(A)+1} out of {len(A)*len(B)}";F=cell(L,J)
			if D is None:D=F
			C.append(F.images[0])
	E=images.image_grid(C,rows=len(B));E=images.draw_grid_annotations(E,C[0].width,C[0].height,H,G);D.images=[E];return D
class Script(scripts.Script):
	def title(A):return'Prompt matrix'
	def ui(A,is_img2img):
		B=False;gr.HTML('<br />')
		with gr.Row():
			with gr.Column():C=gr.Checkbox(label='Put variable parts at start of prompt',value=B,elem_id=A.elem_id('put_at_start'));D=gr.Checkbox(label='Use different seed for each picture',value=B,elem_id=A.elem_id('different_seeds'))
			with gr.Column():E=gr.Radio([_A,_C],label='Select prompt',elem_id=A.elem_id('prompt_type'),value=_A);F=gr.Radio([_B,'space'],label='Select joining char',elem_id=A.elem_id('variations_delimiter'),value=_B)
			with gr.Column():G=gr.Slider(label='Grid margins (px)',minimum=0,maximum=500,value=0,step=2,elem_id=A.elem_id('margin_size'))
		return[C,D,E,F,G]
	def run(N,p,put_at_start,different_seeds,prompt_type,variations_delimiter,margin_size):
		F=variations_delimiter;E=prompt_type;modules.processing.fix_seed(p)
		if E not in[_A,_C]:raise ValueError(f"Unknown prompt type {E}")
		if F not in[_B,'space']:raise ValueError(f"Unknown variations delimiter {F}")
		G=p.prompt if E==_A else p.negative_prompt;I=G[0]if type(G)==list else G;J=p.prompt[0]if type(p.prompt)==list else p.prompt;K=', 'if F==_B else' ';B=[];C=I.split('|');L=2**(len(C)-1)
		for M in range(L):
			D=[B.strip().strip(',')for(A,B)in enumerate(C[1:])if M&1<<A]
			if put_at_start:D=D+[C[0]]
			else:D=[C[0]]+D
			B.append(K.join(D))
		p.n_iter=math.ceil(len(B)/p.batch_size);p.do_not_save_grid=True;print(f"Prompt matrix will create {len(B)} images using a total of {p.n_iter} batches.")
		if E==_A:p.prompt=B
		else:p.negative_prompt=B
		p.seed=[p.seed+(A if different_seeds else 0)for A in range(len(B))];p.prompt_for_display=J;A=process_images(p);H=images.image_grid(A.images,p.batch_size,rows=1<<(len(C)-1)//2);H=images.draw_prompt_matrix(H,A.images[0].width,A.images[0].height,C,margin_size);A.images.insert(0,H);A.index_of_first_image=1;A.infotexts.insert(0,A.infotexts[0])
		if opts.grid_save:images.save_image(A.images[0],p.outpath_grids,'prompt_matrix',extension=opts.grid_format,prompt=I,seed=A.seed,grid=True,p=p)
		return A