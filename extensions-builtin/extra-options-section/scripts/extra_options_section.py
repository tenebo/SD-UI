_C='settingsHintsShowQuicksettings'
_B='choices'
_A=False
import math,gradio as gr
from modules import scripts,shared,ui_components,ui_settings,generation_parameters_copypaste
from modules.ui_components import FormColumn
class ExtraOptionsSection(scripts.Script):
	section='extra_options'
	def __init__(A):A.comps=None;A.setting_names=None
	def title(A):return'Extra options'
	def show(A,is_img2img):return scripts.AlwaysVisible
	def ui(A,is_img2img):
		A.comps=[];A.setting_names=[];A.infotext_fields=[];B=shared.opts.extra_options_img2img if is_img2img else shared.opts.extra_options_txt2img;G={B:A for(A,B)in generation_parameters_copypaste.infotext_to_setting_name_mapping}
		with gr.Blocks()as H:
			with gr.Accordion('Options',open=_A)if shared.opts.extra_options_accordion and B else gr.Group():
				I=math.ceil(len(B)/shared.opts.extra_options_cols)
				for J in range(I):
					with gr.Row():
						for K in range(shared.opts.extra_options_cols):
							D=J*shared.opts.extra_options_cols+K
							if D>=len(B):break
							C=B[D]
							with FormColumn():E=ui_settings.create_setting_component(C)
							A.comps.append(E);A.setting_names.append(C);F=G.get(C)
							if F is not None:A.infotext_fields.append((E,F))
		def L():B=[ui_settings.get_value_for_setting(A)for A in A.setting_names];return B[0]if len(B)==1 else B
		H.load(fn=L,inputs=[],outputs=A.comps,queue=_A,show_progress=_A);return A.comps
	def before_process(B,p,*C):
		for(A,D)in zip(B.setting_names,C):
			if A not in p.override_settings:p.override_settings[A]=D
shared.options_templates.update(shared.options_section(('ui','User interface'),{'extra_options_txt2img':shared.OptionInfo([],'Options in main UI - txt2img',ui_components.DropdownMulti,lambda:{_B:list(shared.opts.data_labels.keys())}).js('info',_C).info('setting entries that also appear in txt2img interfaces').needs_reload_ui(),'extra_options_img2img':shared.OptionInfo([],'Options in main UI - img2img',ui_components.DropdownMulti,lambda:{_B:list(shared.opts.data_labels.keys())}).js('info',_C).info('setting entries that also appear in img2img interfaces').needs_reload_ui(),'extra_options_cols':shared.OptionInfo(1,'Options in main UI - number of columns',gr.Number,{'precision':0}).needs_reload_ui(),'extra_options_accordion':shared.OptionInfo(_A,'Options in main UI - place into an accordion').needs_reload_ui()}))