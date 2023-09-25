_F='n_iter'
_E='sampler_name'
_D='negative_prompt'
_C='prompt'
_B=False
_A=None
import copy,random,shlex,modules.scripts as scripts,gradio as gr
from modules import sd_samplers,errors
from modules.processing import Processed,process_images
from modules.shared import state
def process_string_tag(tag):return tag
def process_int_tag(tag):return int(tag)
def process_float_tag(tag):return float(tag)
def process_boolean_tag(tag):return True if tag=='true'else _B
prompt_tags={'sd_model':_A,'outpath_samples':process_string_tag,'outpath_grids':process_string_tag,'prompt_for_display':process_string_tag,_C:process_string_tag,_D:process_string_tag,'styles':process_string_tag,'seed':process_int_tag,'subseed_strength':process_float_tag,'subseed':process_int_tag,'seed_resize_from_h':process_int_tag,'seed_resize_from_w':process_int_tag,'sampler_index':process_int_tag,_E:process_string_tag,'batch_size':process_int_tag,_F:process_int_tag,'steps':process_int_tag,'cfg_scale':process_float_tag,'width':process_int_tag,'height':process_int_tag,'restore_faces':process_boolean_tag,'tiling':process_boolean_tag,'do_not_save_samples':process_boolean_tag,'do_not_save_grid':process_boolean_tag}
def cmdargs(line):
	B=shlex.split(line);A=0;E={}
	while A<len(B):
		D=B[A];assert D.startswith('--'),f'must start with "--": {D}';assert A+1<len(B),f"missing argument for command line option {D}";C=D[2:]
		if C==_C or C==_D:
			A+=1;F=B[A];A+=1
			while A<len(B)and not B[A].startswith('--'):F+=' ';F+=B[A];A+=1
			E[C]=F;continue
		H=prompt_tags.get(C,_A);assert H,f"unknown commandline option: {D}";G=B[A+1]
		if C==_E:G=sd_samplers.samplers_map.get(G.lower(),_A)
		E[C]=H(G);A+=2
	return E
def load_prompt_file(file):
	if file is _A:return _A,gr.update(),gr.update(lines=7)
	else:A=[A.strip()for A in file.decode('utf8',errors='ignore').split('\n')];return _A,'\n'.join(A),gr.update(lines=7)
class Script(scripts.Script):
	def title(A):return'Prompts from file or textbox'
	def ui(B,is_img2img):D=gr.Checkbox(label='Iterate seed every line',value=_B,elem_id=B.elem_id('checkbox_iterate'));E=gr.Checkbox(label='Use same random seed for all lines',value=_B,elem_id=B.elem_id('checkbox_iterate_batch'));A=gr.Textbox(label='List of prompt inputs',lines=1,elem_id=B.elem_id('prompt_txt'));C=gr.File(label='Upload prompt inputs',type='binary',elem_id=B.elem_id('file'));C.change(fn=load_prompt_file,inputs=[C],outputs=[C,A,A],show_progress=_B);A.change(lambda tb:gr.update(lines=7)if'\n'in tb else gr.update(lines=2),inputs=[A],outputs=[A],show_progress=_B);return[D,E,A]
	def run(N,p,checkbox_iterate,checkbox_iterate_batch,prompt_txt):
		E=checkbox_iterate;F=[A for A in(A.strip()for A in prompt_txt.splitlines())if A];p.do_not_save_grid=True;C=0;G=[]
		for B in F:
			if'--'in B:
				try:A=cmdargs(B)
				except Exception:errors.report(f"Error parsing line {B} as commandline",exc_info=True);A={_C:B}
			else:A={_C:B}
			C+=A.get(_F,p.n_iter);G.append(A)
		print(f"Will process {len(F)} lines in {C} jobs.")
		if(E or checkbox_iterate_batch)and p.seed==-1:p.seed=int(random.randrange(4294967294))
		state.job_count=C;H=[];I=[];J=[]
		for A in G:
			state.job=f"{state.job_no+1} out of {state.job_count}";K=copy.copy(p)
			for(L,M)in A.items():setattr(K,L,M)
			D=process_images(K);H+=D.images
			if E:p.seed=p.seed+p.batch_size*p.n_iter
			I+=D.all_prompts;J+=D.infotexts
		return Processed(p,H,p.seed,'',all_prompts=I,infotexts=J)