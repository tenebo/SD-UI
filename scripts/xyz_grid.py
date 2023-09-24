_U='Z Values'
_T='Z Type'
_S='Y Values'
_R='Y Type'
_Q='X Values'
_P='X Type'
_O='sampler_name'
_N='Sampler'
_M='uni_pc_order'
_L='CLIP_stop_at_last_layers'
_K='None'
_J='Hires steps'
_I='Steps'
_H=', '
_G='Var. seed'
_F='Seed'
_E='Nothing'
_D='y'
_C=False
_B=True
_A=None
from collections import namedtuple
from copy import copy
from itertools import permutations,chain
import random,csv,os.path
from io import StringIO
from PIL import Image
import numpy as np,modules.scripts as scripts,gradio as gr
from modules import images,sd_samplers,processing,sd_models,sd_vae,sd_samplers_kdiffusion,errors
from modules.processing import process_images,Processed,StandardDemoProcessingTxt2Img
from modules.shared import opts,state
import modules.shared as shared,modules.sd_samplers,modules.sd_models,modules.sd_vae,re
from modules.ui_components import ToolButton
fill_values_symbol='📒'
AxisInfo=namedtuple('AxisInfo',['axis','values'])
def apply_field(field):
	def A(p,x,xs):setattr(p,field,x)
	return A
def apply_prompt(p,x,xs):
	A=xs
	if A[0]not in p.prompt and A[0]not in p.negative_prompt:raise RuntimeError(f"Prompt S/R did not find {A[0]} in prompt or negative prompt.")
	p.prompt=p.prompt.replace(A[0],x);p.negative_prompt=p.negative_prompt.replace(A[0],x)
def apply_order(p,x,xs):
	B=[]
	for A in x:B.append((p.prompt.find(A),A))
	B.sort(key=lambda t:t[0]);D=[]
	for(H,A)in B:E=p.prompt.find(A);D.append(p.prompt[0:E]);p.prompt=p.prompt[E+len(A):]
	C=''
	for(F,G)in enumerate(D):C+=G;C+=x[F]
	p.prompt=C+p.prompt
def confirm_samplers(p,xs):
	for A in xs:
		if A.lower()not in sd_samplers.samplers_map:raise RuntimeError(f"Unknown sampler: {A}")
def apply_checkpoint(p,x,xs):
	A=modules.sd_models.get_closet_checkpoint_match(x)
	if A is _A:raise RuntimeError(f"Unknown checkpoint: {x}")
	p.override_settings['sd_model_checkpoint']=A.name
def confirm_checkpoints(p,xs):
	for A in xs:
		if modules.sd_models.get_closet_checkpoint_match(A)is _A:raise RuntimeError(f"Unknown checkpoint: {A}")
def confirm_checkpoints_or_none(p,xs):
	for A in xs:
		if A in(_A,'',_K,'none'):continue
		if modules.sd_models.get_closet_checkpoint_match(A)is _A:raise RuntimeError(f"Unknown checkpoint: {A}")
def apply_clip_skip(p,x,xs):opts.data[_L]=x
def apply_upscale_latent_space(p,x,xs):
	A='use_scale_latent_for_hires_fix'
	if x.lower().strip()!='0':opts.data[A]=_B
	else:opts.data[A]=_C
def find_vae(name):
	A=name
	if A.lower()in['auto','automatic']:return modules.sd_vae.unspecified
	if A.lower()=='none':return
	else:
		B=[B for B in sorted(modules.sd_vae.vae_dict,key=lambda x:len(x))if A.lower().strip()in B.lower()]
		if len(B)==0:print(f"No VAE found for {A}; using automatic");return modules.sd_vae.unspecified
		else:return modules.sd_vae.vae_dict[B[0]]
def apply_vae(p,x,xs):modules.sd_vae.reload_vae_weights(shared.sd_model,vae_file=find_vae(x))
def apply_styles(p,x,_):p.styles.extend(x.split(','))
def apply_uni_pc_order(p,x,xs):opts.data[_M]=min(x,p.steps-1)
def apply_face_restore(p,opt,x):
	A=opt;A=A.lower()
	if A=='codeformer':B=_B;p.face_restoration_model='CodeFormer'
	elif A=='gfpgan':B=_B;p.face_restoration_model='GFPGAN'
	else:B=A in('true','yes',_D,'1')
	p.restore_faces=B
def apply_override(field,boolean=_C):
	def A(p,x,xs):
		if boolean:x=_B if x.lower()=='true'else _C
		p.override_settings[field]=x
	return A
def boolean_choice(reverse=_C):
	def A():A='True';B='False';return[B,A]if reverse else[A,B]
	return A
def format_value_add_label(p,opt,x):
	if type(x)==float:x=round(x,8)
	return f"{opt.label}: {x}"
def format_value(p,opt,x):
	if type(x)==float:x=round(x,8)
	return x
def format_value_join_list(p,opt,x):return _H.join(x)
def do_nothing(p,x,xs):0
def format_nothing(p,opt,x):return''
def format_remove_path(p,opt,x):return os.path.basename(x)
def str_permutations(x):"dummy function for specifying it in AxisOption's type when you want to get a list of permutations";return x
def list_to_csv_string(data_list):
	with StringIO()as A:csv.writer(A).writerow(data_list);return A.getvalue().strip()
def csv_string_to_list_strip(data_str):return list(map(str.strip,chain.from_iterable(csv.reader(StringIO(data_str)))))
class AxisOption:
	def __init__(A,label,type,apply,format_value=format_value_add_label,confirm=_A,cost=.0,choices=_A):A.label=label;A.type=type;A.apply=apply;A.format_value=format_value;A.confirm=confirm;A.cost=cost;A.choices=choices
class AxisOptionImg2Img(AxisOption):
	def __init__(A,*B,**C):super().__init__(*B,**C);A.is_img2img=_B
class AxisOptionTxt2Img(AxisOption):
	def __init__(A,*B,**C):super().__init__(*B,**C);A.is_img2img=_C
axis_options=[AxisOption(_E,str,do_nothing,format_value=format_nothing),AxisOption(_F,int,apply_field('seed')),AxisOption(_G,int,apply_field('subseed')),AxisOption('Var. strength',float,apply_field('subseed_strength')),AxisOption(_I,int,apply_field('steps')),AxisOptionTxt2Img(_J,int,apply_field('hr_second_pass_steps')),AxisOption('CFG Scale',float,apply_field('cfg_scale')),AxisOptionImg2Img('Image CFG Scale',float,apply_field('image_cfg_scale')),AxisOption('Prompt S/R',str,apply_prompt,format_value=format_value),AxisOption('Prompt order',str_permutations,apply_order,format_value=format_value_join_list),AxisOptionTxt2Img(_N,str,apply_field(_O),format_value=format_value,confirm=confirm_samplers,choices=lambda:[A.name for A in sd_samplers.samplers if A.name not in opts.hide_samplers]),AxisOptionTxt2Img('Hires sampler',str,apply_field('hr_sampler_name'),confirm=confirm_samplers,choices=lambda:[A.name for A in sd_samplers.samplers_for_img2img if A.name not in opts.hide_samplers]),AxisOptionImg2Img(_N,str,apply_field(_O),format_value=format_value,confirm=confirm_samplers,choices=lambda:[A.name for A in sd_samplers.samplers_for_img2img if A.name not in opts.hide_samplers]),AxisOption('Checkpoint name',str,apply_checkpoint,format_value=format_remove_path,confirm=confirm_checkpoints,cost=1.,choices=lambda:sorted(sd_models.checkpoints_list,key=str.casefold)),AxisOption('Negative Guidance minimum sigma',float,apply_field('s_min_uncond')),AxisOption('Sigma Churn',float,apply_field('s_churn')),AxisOption('Sigma min',float,apply_field('s_tmin')),AxisOption('Sigma max',float,apply_field('s_tmax')),AxisOption('Sigma noise',float,apply_field('s_noise')),AxisOption('Schedule type',str,apply_override('k_sched_type'),choices=lambda:list(sd_samplers_kdiffusion.k_diffusion_scheduler)),AxisOption('Schedule min sigma',float,apply_override('sigma_min')),AxisOption('Schedule max sigma',float,apply_override('sigma_max')),AxisOption('Schedule rho',float,apply_override('rho')),AxisOption('Eta',float,apply_field('eta')),AxisOption('Clip skip',int,apply_clip_skip),AxisOption('Denoising',float,apply_field('denoising_strength')),AxisOption('Initial noise multiplier',float,apply_field('initial_noise_multiplier')),AxisOption('Extra noise',float,apply_override('img2img_extra_noise')),AxisOptionTxt2Img('Hires upscaler',str,apply_field('hr_upscaler'),choices=lambda:[*shared.latent_upscale_modes,*[A.name for A in shared.sd_upscalers]]),AxisOptionImg2Img('Cond. Image Mask Weight',float,apply_field('inpainting_mask_weight')),AxisOption('VAE',str,apply_vae,cost=.7,choices=lambda:[_K]+list(sd_vae.vae_dict)),AxisOption('Styles',str,apply_styles,choices=lambda:list(shared.prompt_styles.styles)),AxisOption('UniPC Order',int,apply_uni_pc_order,cost=.5),AxisOption('Face restore',str,apply_face_restore,format_value=format_value),AxisOption('Token merging ratio',float,apply_override('token_merging_ratio')),AxisOption('Token merging ratio high-res',float,apply_override('token_merging_ratio_hr')),AxisOption('Always discard next-to-last sigma',str,apply_override('always_discard_next_to_last_sigma',boolean=_B),choices=boolean_choice(reverse=_B)),AxisOption('SGM noise multiplier',str,apply_override('sgm_noise_multiplier',boolean=_B),choices=boolean_choice(reverse=_B)),AxisOption('Refiner checkpoint',str,apply_field('refiner_checkpoint'),format_value=format_remove_path,confirm=confirm_checkpoints_or_none,cost=1.,choices=lambda:[_K]+sorted(sd_models.checkpoints_list,key=str.casefold)),AxisOption('Refiner switch at',float,apply_field('refiner_switch_at')),AxisOption('RNG source',str,apply_override('randn_source'),choices=lambda:['GPU','CPU','NV'])]
def draw_xyz_grid(p,xs,ys,zs,x_labels,y_labels,z_labels,cell,draw_legend,include_lone_images,include_sub_grids,first_axes_processed,second_axes_processed,margin_size):
	S=draw_legend;O=second_axes_processed;P=first_axes_processed;J=zs;B=ys;C=xs;V=[[images.GridAnnotation(A)]for A in x_labels];W=[[images.GridAnnotation(A)]for A in y_labels];X=[[images.GridAnnotation(A)]for A in z_labels];L=len(C)*len(B)*len(J);A=_A;state.job_count=L*p.n_iter
	def M(x,y,z,ix,iy,iz):
		nonlocal A
		def F(ix,iy,iz):return ix+iy*len(C)+iz*len(C)*len(B)
		state.job=f"{F(ix,iy,iz)+1} out of {L}";D=cell(x,y,z,ix,iy,iz)
		if A is _A:A=copy(D);A.images=[_A]*L;A.all_prompts=[_A]*L;A.all_seeds=[_A]*L;A.infotexts=[_A]*L;A.index_of_first_image=1
		E=F(ix,iy,iz)
		if D.images:A.images[E]=D.images[0];A.all_prompts[E]=D.prompt;A.all_seeds[E]=D.seed;A.infotexts[E]=D.infotexts[0]
		else:
			G='P';H=A.width,A.height
			if A.images[0]is not _A:G=A.images[0].mode;H=A.images[0].size
			A.images[E]=Image.new(G,H)
	if P=='x':
		for(D,E)in enumerate(C):
			if O==_D:
				for(F,G)in enumerate(B):
					for(H,I)in enumerate(J):M(E,G,I,D,F,H)
			else:
				for(H,I)in enumerate(J):
					for(F,G)in enumerate(B):M(E,G,I,D,F,H)
	elif P==_D:
		for(F,G)in enumerate(B):
			if O=='x':
				for(D,E)in enumerate(C):
					for(H,I)in enumerate(J):M(E,G,I,D,F,H)
			else:
				for(H,I)in enumerate(J):
					for(D,E)in enumerate(C):M(E,G,I,D,F,H)
	elif P=='z':
		for(H,I)in enumerate(J):
			if O=='x':
				for(D,E)in enumerate(C):
					for(F,G)in enumerate(B):M(E,G,I,D,F,H)
			else:
				for(F,G)in enumerate(B):
					for(D,E)in enumerate(C):M(E,G,I,D,F,H)
	if not A:print('Unexpected error: Processing could not begin, you may need to refresh the tab or restart the service.');return Processed(p,[])
	elif not any(A.images):print('Unexpected error: draw_xyz_grid failed to return even a single processed image');return Processed(p,[])
	T=len(J)
	for N in range(T):
		K=N*len(C)*len(B)+N;Y=K+len(C)*len(B);Q=images.image_grid(A.images[K:Y],rows=len(B))
		if S:Q=images.draw_grid_annotations(Q,A.images[K].size[0],A.images[K].size[1],V,W,margin_size)
		A.images.insert(N,Q);A.all_prompts.insert(N,A.all_prompts[K]);A.all_seeds.insert(N,A.all_seeds[K]);A.infotexts.insert(N,A.infotexts[K])
	U=A.images[0].size;R=images.image_grid(A.images[:T],rows=1)
	if S:R=images.draw_grid_annotations(R,U[0],U[1],X,[[images.GridAnnotation()]])
	A.images.insert(0,R);A.infotexts.insert(0,A.infotexts[0]);return A
class SharedSettingsStackHelper:
	def __enter__(A):A.CLIP_stop_at_last_layers=opts.CLIP_stop_at_last_layers;A.vae=opts.sd_vae;A.uni_pc_order=opts.uni_pc_order
	def __exit__(A,exc_type,exc_value,tb):opts.data['sd_vae']=A.vae;opts.data[_M]=A.uni_pc_order;modules.sd_models.reload_model_weights();modules.sd_vae.reload_vae_weights();opts.data[_L]=A.CLIP_stop_at_last_layers
re_range=re.compile('\\s*([+-]?\\s*\\d+)\\s*-\\s*([+-]?\\s*\\d+)(?:\\s*\\(([+-]\\d+)\\s*\\))?\\s*')
re_range_float=re.compile('\\s*([+-]?\\s*\\d+(?:.\\d*)?)\\s*-\\s*([+-]?\\s*\\d+(?:.\\d*)?)(?:\\s*\\(([+-]\\d+(?:.\\d*)?)\\s*\\))?\\s*')
re_range_count=re.compile('\\s*([+-]?\\s*\\d+)\\s*-\\s*([+-]?\\s*\\d+)(?:\\s*\\[(\\d+)\\s*])?\\s*')
re_range_count_float=re.compile('\\s*([+-]?\\s*\\d+(?:.\\d*)?)\\s*-\\s*([+-]?\\s*\\d+(?:.\\d*)?)(?:\\s*\\[(\\d+(?:.\\d*)?)\\s*])?\\s*')
class Script(scripts.Script):
	def title(A):return'X/Y/Z plot'
	def ui(A,is_img2img):
		T='compact';U='Z values';V='Y values';W='X values';M='index';A.current_axis_options=[A for A in axis_options if type(A)==AxisOption or A.is_img2img==is_img2img]
		with gr.Row():
			with gr.Column(scale=19):
				with gr.Row():I=gr.Dropdown(label='X type',choices=[A.label for A in A.current_axis_options],value=A.current_axis_options[1].label,type=M,elem_id=A.elem_id('x_type'));B=gr.Textbox(label=W,lines=1,elem_id=A.elem_id('x_values'));C=gr.Dropdown(label=W,visible=_C,multiselect=_B,interactive=_B);N=ToolButton(value=fill_values_symbol,elem_id='xyz_grid_fill_x_tool_button',visible=_C)
				with gr.Row():J=gr.Dropdown(label='Y type',choices=[A.label for A in A.current_axis_options],value=A.current_axis_options[0].label,type=M,elem_id=A.elem_id('y_type'));D=gr.Textbox(label=V,lines=1,elem_id=A.elem_id('y_values'));E=gr.Dropdown(label=V,visible=_C,multiselect=_B,interactive=_B);O=ToolButton(value=fill_values_symbol,elem_id='xyz_grid_fill_y_tool_button',visible=_C)
				with gr.Row():K=gr.Dropdown(label='Z type',choices=[A.label for A in A.current_axis_options],value=A.current_axis_options[0].label,type=M,elem_id=A.elem_id('z_type'));F=gr.Textbox(label=U,lines=1,elem_id=A.elem_id('z_values'));G=gr.Dropdown(label=U,visible=_C,multiselect=_B,interactive=_B);P=ToolButton(value=fill_values_symbol,elem_id='xyz_grid_fill_z_tool_button',visible=_C)
		with gr.Row(variant=T,elem_id='axis_options'):
			with gr.Column():a=gr.Checkbox(label='Draw legend',value=_B,elem_id=A.elem_id('draw_legend'));b=gr.Checkbox(label='Keep -1 for seeds',value=_C,elem_id=A.elem_id('no_fixed_seeds'))
			with gr.Column():c=gr.Checkbox(label='Include Sub Images',value=_C,elem_id=A.elem_id('include_lone_images'));d=gr.Checkbox(label='Include Sub Grids',value=_C,elem_id=A.elem_id('include_sub_grids'))
			with gr.Column():e=gr.Slider(label='Grid margins (px)',minimum=0,maximum=500,value=0,step=2,elem_id=A.elem_id('margin_size'))
			with gr.Column():H=gr.Checkbox(label='Use text inputs instead of dropdowns',value=_C,elem_id=A.elem_id('csv_mode'))
		with gr.Row(variant=T,elem_id='swap_axes'):f=gr.Button(value='Swap X/Y axes',elem_id='xy_grid_swap_axes_button');g=gr.Button(value='Swap Y/Z axes',elem_id='yz_grid_swap_axes_button');h=gr.Button(value='Swap X/Z axes',elem_id='xz_grid_swap_axes_button')
		def Q(axis1_type,axis1_values,axis1_values_dropdown,axis2_type,axis2_values,axis2_values_dropdown):return A.current_axis_options[axis2_type].label,axis2_values,axis2_values_dropdown,A.current_axis_options[axis1_type].label,axis1_values,axis1_values_dropdown
		X=[I,B,C,J,D,E];f.click(Q,inputs=X,outputs=X);Y=[J,D,E,K,F,G];g.click(Q,inputs=Y,outputs=Y);Z=[I,B,C,K,F,G];h.click(Q,inputs=Z,outputs=Z)
		def R(axis_type,csv_mode):
			B=A.current_axis_options[axis_type]
			if B.choices:
				if csv_mode:return list_to_csv_string(B.choices()),gr.update()
				else:return gr.update(),B.choices()
			else:return gr.update(),gr.update()
		N.click(fn=R,inputs=[I,H],outputs=[B,C]);O.click(fn=R,inputs=[J,H],outputs=[D,E]);P.click(fn=R,inputs=[K,H],outputs=[F,G])
		def L(axis_type,axis_values,axis_values_dropdown,csv_mode):
			F=csv_mode;C=axis_values_dropdown;D=axis_values;B=A.current_axis_options[axis_type].choices;E=B is not _A
			if E:
				B=B()
				if F:
					if C:D=list_to_csv_string(list(filter(lambda x:x in B,C)));C=[]
				elif D:C=list(filter(lambda x:x in B,csv_string_to_list_strip(D)));D=''
			return gr.Button.update(visible=E),gr.Textbox.update(visible=not E or F,value=D),gr.update(choices=B if E else _A,visible=E and not F,value=C)
		I.change(fn=L,inputs=[I,B,C,H],outputs=[N,B,C]);J.change(fn=L,inputs=[J,D,E,H],outputs=[O,D,E]);K.change(fn=L,inputs=[K,F,G,H],outputs=[P,F,G])
		def i(csv_mode,x_type,x_values,x_values_dropdown,y_type,y_values,y_values_dropdown,z_type,z_values,z_values_dropdown):A=csv_mode;B,C,D=L(x_type,x_values,x_values_dropdown,A);E,F,G=L(y_type,y_values,y_values_dropdown,A);H,I,J=L(z_type,z_values,z_values_dropdown,A);return B,C,D,E,F,G,H,I,J
		H.change(fn=i,inputs=[H,I,B,C,J,D,E,K,F,G],outputs=[N,B,C,O,D,E,P,F,G])
		def S(axis,params):A=f"{axis} Values";B=params.get(A,'');C=csv_string_to_list_strip(B);return gr.update(value=C)
		A.infotext_fields=(I,_P),(B,_Q),(C,lambda params:S('X',params)),(J,_R),(D,_S),(E,lambda params:S('Y',params)),(K,_T),(F,_U),(G,lambda params:S('Z',params));return[I,B,C,J,D,E,K,F,G,a,c,d,b,e,H]
	def run(N,p,x_type,x_values,x_values_dropdown,y_type,y_values,y_values_dropdown,z_type,z_values,z_values_dropdown,draw_legend,include_lone_images,include_sub_grids,no_fixed_seeds,margin_size,csv_mode):
		W=include_sub_grids;X=include_lone_images;Y=z_values_dropdown;Z=y_values_dropdown;a=x_values_dropdown;Q=z_values;R=y_values;S=x_values;O=csv_mode;K=no_fixed_seeds
		if not K:modules.processing.fix_seed(p)
		if not opts.return_grid:p.batch_size=1
		def T(opt,vals,vals_dropdown):
			E=opt
			if E.label==_E:return[0]
			if E.choices is not _A and not O:A=vals_dropdown
			else:A=csv_string_to_list_strip(vals)
			if E.type==int:
				D=[]
				for F in A:
					B=re_range.fullmatch(F);C=re_range_count.fullmatch(F)
					if B is not _A:G=int(B.group(1));H=int(B.group(2))+1;I=int(B.group(3))if B.group(3)is not _A else 1;D+=list(range(G,H,I))
					elif C is not _A:G=int(C.group(1));H=int(C.group(2));J=int(C.group(3))if C.group(3)is not _A else 1;D+=[int(A)for A in np.linspace(start=G,stop=H,num=J).tolist()]
					else:D.append(F)
				A=D
			elif E.type==float:
				D=[]
				for F in A:
					B=re_range_float.fullmatch(F);C=re_range_count_float.fullmatch(F)
					if B is not _A:G=float(B.group(1));H=float(B.group(2));I=float(B.group(3))if B.group(3)is not _A else 1;D+=np.arange(G,H+I,I).tolist()
					elif C is not _A:G=float(C.group(1));H=float(C.group(2));J=int(C.group(3))if C.group(3)is not _A else 1;D+=np.linspace(start=G,stop=H,num=J).tolist()
					else:D.append(F)
				A=D
			elif E.type==str_permutations:A=list(permutations(A))
			A=[E.type(A)for A in A]
			if E.confirm:E.confirm(p,A)
			return A
		C=N.current_axis_options[x_type]
		if C.choices is not _A and not O:S=list_to_csv_string(a)
		D=T(C,S,a);E=N.current_axis_options[y_type]
		if E.choices is not _A and not O:R=list_to_csv_string(Z)
		F=T(E,R,Z);G=N.current_axis_options[z_type]
		if G.choices is not _A and not O:Q=list_to_csv_string(Y)
		B=T(G,Q,Y);Image.MAX_IMAGE_PIXELS=_A;b=round(len(D)*len(F)*len(B)*p.width*p.height/1000000);assert b<opts.img_max_size_mp,f"Error: Resulting grid would be too large ({b} MPixels) (max configured size is {opts.img_max_size_mp} MPixels)"
		def U(axis_opt,axis_list):
			A=axis_list
			if axis_opt.label in[_F,_G]:return[int(random.randrange(4294967294))if A is _A or A==''or A==-1 else A for A in A]
			else:return A
		if not K:D=U(C,D);F=U(E,F);B=U(G,B)
		if C.label==_I:H=sum(D)*len(F)*len(B)
		elif E.label==_I:H=sum(F)*len(D)*len(B)
		elif G.label==_I:H=sum(B)*len(D)*len(F)
		else:H=p.steps*len(D)*len(F)*len(B)
		if isinstance(p,StandardDemoProcessingTxt2Img)and p.enable_hr:
			if C.label==_J:H+=sum(D)*len(F)*len(B)
			elif E.label==_J:H+=sum(F)*len(D)*len(B)
			elif G.label==_J:H+=sum(B)*len(D)*len(F)
			elif p.hr_second_pass_steps:H+=p.hr_second_pass_steps*len(D)*len(F)*len(B)
			else:H*=2
		H*=p.n_iter;V=p.n_iter*p.batch_size;d=f"; {V} images per cell"if V>1 else'';e='s'if len(B)>1 else'';print(f"X/Y/Z plot will create {len(D)*len(F)*len(B)*V} images on {len(B)} {len(D)}x{len(F)} grid{e}{d}. (Total steps to process: {H})");shared.total_tqdm.updateTotal(H);state.xyz_plot_x=AxisInfo(C,D);state.xyz_plot_y=AxisInfo(E,F);state.xyz_plot_z=AxisInfo(G,B);P='z';I=_D
		if C.cost>E.cost and C.cost>G.cost:
			P='x'
			if E.cost>G.cost:I=_D
			else:I='z'
		elif E.cost>C.cost and E.cost>G.cost:
			P=_D
			if C.cost>G.cost:I='x'
			else:I='z'
		elif G.cost>C.cost and G.cost>E.cost:
			P='z'
			if C.cost>E.cost:I='x'
			else:I=_D
		L=[_A]*(1+len(B))
		def f(x,y,z,ix,iy,iz):
			if shared.state.interrupted:return Processed(p,[],p.seed,'')
			A=copy(p);A.styles=A.styles[:];C.apply(A,x,D);E.apply(A,y,F);G.apply(A,z,B)
			try:H=process_images(A)
			except Exception as J:errors.display(J,'generating image for xyz plot');H=Processed(p,[],p.seed,'')
			I=1+iz
			if L[I]is _A and ix==0 and iy==0:
				A.extra_generation_params=copy(A.extra_generation_params);A.extra_generation_params['Script']=N.title()
				if C.label!=_E:
					A.extra_generation_params[_P]=C.label;A.extra_generation_params[_Q]=S
					if C.label in[_F,_G]and not K:A.extra_generation_params['Fixed X Values']=_H.join([str(A)for A in D])
				if E.label!=_E:
					A.extra_generation_params[_R]=E.label;A.extra_generation_params[_S]=R
					if E.label in[_F,_G]and not K:A.extra_generation_params['Fixed Y Values']=_H.join([str(A)for A in F])
				L[I]=processing.create_infotext(A,A.all_prompts,A.all_seeds,A.all_subseeds)
			if L[0]is _A and ix==0 and iy==0 and iz==0:
				A.extra_generation_params=copy(A.extra_generation_params)
				if G.label!=_E:
					A.extra_generation_params[_T]=G.label;A.extra_generation_params[_U]=Q
					if G.label in[_F,_G]and not K:A.extra_generation_params['Fixed Z Values']=_H.join([str(A)for A in B])
				L[0]=processing.create_infotext(A,A.all_prompts,A.all_seeds,A.all_subseeds)
			return H
		with SharedSettingsStackHelper():A=draw_xyz_grid(p,xs=D,ys=F,zs=B,x_labels=[C.format_value(p,C,A)for A in D],y_labels=[E.format_value(p,E,A)for A in F],z_labels=[G.format_value(p,G,A)for A in B],cell=f,draw_legend=draw_legend,include_lone_images=X,include_sub_grids=W,first_axes_processed=P,second_axes_processed=I,margin_size=margin_size)
		if not A.images:return A
		J=len(B);A.infotexts[:1+J]=L[:1+J]
		if not X:A.images=A.images[:J+1]
		if opts.grid_save:
			g=J+1 if J>1 else 1
			for M in range(g):c=M-1 if M>0 else M;images.save_image(A.images[M],p.outpath_grids,'xyz_grid',info=A.infotexts[M],extension=opts.grid_format,prompt=A.all_prompts[c],seed=A.all_seeds[c],grid=_B,p=A)
		if not W:
			for h in range(J):del A.images[1];del A.all_prompts[1];del A.all_seeds[1];del A.infotexts[1]
		return A