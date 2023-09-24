_E='subseed_strength'
_D='Seed'
_C='seed'
_B=None
_A=False
import json,gradio as gr
from modules import scripts,ui,errors
from modules.shared import cmd_opts
from modules.ui_components import ToolButton
class ScriptSeed(scripts.ScriptBuiltinUI):
	section=_C;create_group=_A
	def __init__(A):A.seed=_B;A.reuse_seed=_B;A.reuse_subseed=_B
	def title(A):return _D
	def show(A,is_img2img):return scripts.AlwaysVisible
	def ui(A,is_img2img):
		E='Seed resize from-1';F="')}";G="function(){setRandomSeed('";H='subseed';C='Variation seed'
		with gr.Row(elem_id=A.elem_id('seed_row')):
			if cmd_opts.use_textbox_seed:A.seed=gr.Textbox(label=_D,value='',elem_id=A.elem_id(_C),min_width=100)
			else:A.seed=gr.Number(label=_D,value=-1,elem_id=A.elem_id(_C),min_width=100,precision=0)
			L=ToolButton(ui.random_symbol,elem_id=A.elem_id('random_seed'),label='Random seed');M=ToolButton(ui.reuse_symbol,elem_id=A.elem_id('reuse_seed'),label='Reuse seed');B=gr.Checkbox(label='Extra',elem_id=A.elem_id('subseed_show'),value=_A)
		with gr.Group(visible=_A,elem_id=A.elem_id('seed_extras'))as N:
			with gr.Row(elem_id=A.elem_id('subseed_row')):D=gr.Number(label=C,value=-1,elem_id=A.elem_id(H),precision=0);O=ToolButton(ui.random_symbol,elem_id=A.elem_id('random_subseed'));P=ToolButton(ui.reuse_symbol,elem_id=A.elem_id('reuse_subseed'));I=gr.Slider(label='Variation strength',value=.0,minimum=0,maximum=1,step=.01,elem_id=A.elem_id(_E))
			with gr.Row(elem_id=A.elem_id('seed_resize_from_row')):J=gr.Slider(minimum=0,maximum=2048,step=8,label='Resize seed from width',value=0,elem_id=A.elem_id('seed_resize_from_w'));K=gr.Slider(minimum=0,maximum=2048,step=8,label='Resize seed from height',value=0,elem_id=A.elem_id('seed_resize_from_h'))
		L.click(fn=_B,_js=G+A.elem_id(_C)+F,show_progress=_A,inputs=[],outputs=[]);O.click(fn=_B,_js=G+A.elem_id(H)+F,show_progress=_A,inputs=[],outputs=[]);B.change(lambda x:gr.update(visible=x),show_progress=_A,inputs=[B],outputs=[N]);A.infotext_fields=[(A.seed,_D),(B,lambda d:C in d or E in d),(D,C),(I,'Variation seed strength'),(J,E),(K,'Seed resize from-2')];A.on_after_component(lambda x:connect_reuse_seed(A.seed,M,x.component,_A),elem_id=f"generation_info_{A.tabname}");A.on_after_component(lambda x:connect_reuse_seed(D,P,x.component,True),elem_id=f"generation_info_{A.tabname}");return A.seed,B,D,I,J,K
	def setup(E,p,seed,seed_checkbox,subseed,subseed_strength,seed_resize_from_w,seed_resize_from_h):
		A=seed_resize_from_h;B=seed_resize_from_w;C=subseed_strength;D=seed_checkbox;p.seed=seed
		if D and C>0:p.subseed=subseed;p.subseed_strength=C
		if D and B>0 and A>0:p.seed_resize_from_w=B;p.seed_resize_from_h=A
def connect_reuse_seed(seed,reuse_seed,generation_info,is_subseed):
	" Connects a 'reuse (sub)seed' button's click event so that it copies last used\n        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength\n        was 0, i.e. no variation seed was used, it copies the normal seed value instead.";A=seed
	def B(gen_info_string,index):
		C=gen_info_string;A=index;D=-1
		try:
			B=json.loads(C);A-=B.get('index_of_first_image',0)
			if is_subseed and B.get(_E,0)>0:E=B.get('all_subseeds',[-1]);D=E[A if 0<=A<len(E)else 0]
			else:F=B.get('all_seeds',[-1]);D=F[A if 0<=A<len(F)else 0]
		except json.decoder.JSONDecodeError:
			if C:errors.report(f"Error parsing JSON generation info: {C}")
		return[D,gr.update()]
	reuse_seed.click(fn=B,_js='(x, y) => [x, selected_gallery_index()]',show_progress=_A,inputs=[generation_info,A],outputs=[A,A])