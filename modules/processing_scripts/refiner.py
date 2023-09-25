_B='Refiner'
_A=None
import gradio as gr
from modules import scripts,sd_models
from modules.ui_common import create_refresh_button
from modules.ui_components import InputAccordion
class ScriptRefiner(scripts.ScriptBuiltinUI):
	section='accordions';create_group=False
	def __init__(A):0
	def title(A):return _B
	def show(A,is_img2img):return scripts.AlwaysVisible
	def ui(A,is_img2img):
		with InputAccordion(False,label=_B,elem_id=A.elem_id('enable'))as C:
			with gr.Row():B=gr.Dropdown(label='Checkpoint',elem_id=A.elem_id('checkpoint'),choices=sd_models.checkpoint_tiles(),value='',tooltip='switch to another model in the middle of generation');create_refresh_button(B,sd_models.list_models,lambda:{'choices':sd_models.checkpoint_tiles()},A.elem_id('checkpoint_refresh'));D=gr.Slider(value=.8,label='Switch at',minimum=.01,maximum=1.,step=.01,elem_id=A.elem_id('switch_at'),tooltip='fraction of sampling steps when the switch to refiner model should happen; 1=never, 0.5=switch in the middle of generation')
		def E(title):A=sd_models.get_closet_checkpoint_match(title);return _A if A is _A else A.title
		A.infotext_fields=[(C,lambda d:_B in d),(B,lambda d:E(d.get(_B))),(D,'Refiner switch at')];return C,B,D
	def setup(B,p,enable_refiner,refiner_checkpoint,refiner_switch_at):
		A=refiner_checkpoint
		if not enable_refiner or A in(_A,'','None'):p.refiner_checkpoint=_A;p.refiner_switch_at=_A
		else:p.refiner_checkpoint=A;p.refiner_switch_at=refiner_switch_at