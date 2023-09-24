_D='compact'
_C='quicksettings'
_B=False
_A=None
import gradio as gr
from modules import ui_common,shared,script_callbacks,scripts,sd_models,sysinfo
from modules.call_queue import wrap_gradio_call
from modules.shared import opts
from modules.ui_components import FormRow
from modules.ui_gradio_extensions import reload_javascript
def get_value_for_setting(key):C=getattr(opts,key);A=opts.data_labels[key];B=A.component_args()if callable(A.component_args)else A.component_args or{};B={A:B for(A,B)in B.items()if A not in{'precision'}};return gr.update(value=C,**B)
def create_setting_component(key,is_quicksettings=_B):
	B=key
	def F():return opts.data[B]if B in opts.data else opts.data_labels[B].default
	A=opts.data_labels[B];E=type(A.default);G=A.component_args()if callable(A.component_args)else A.component_args
	if A.component is not _A:C=A.component
	elif E==str:C=gr.Textbox
	elif E==int:C=gr.Number
	elif E==bool:C=gr.Checkbox
	else:raise Exception(f"bad options item type: {E} for key {B}")
	H=f"setting_{B}"
	if A.refresh is not _A:
		if is_quicksettings:D=C(label=A.label,value=F(),elem_id=H,**G or{});ui_common.create_refresh_button(D,A.refresh,A.component_args,f"refresh_{B}")
		else:
			with FormRow():D=C(label=A.label,value=F(),elem_id=H,**G or{});ui_common.create_refresh_button(D,A.refresh,A.component_args,f"refresh_{B}")
	else:D=C(label=A.label,value=F(),elem_id=H,**G or{})
	return D
class UiSettings:
	submit=_A;result=_A;interface=_A;components=_A;component_dict=_A;dummy_component=_A;quicksettings_list=_A;quicksettings_names=_A;text_settings=_A
	def run_settings(C,*F):
		A=[]
		for(B,D,E)in zip(opts.data_labels.keys(),F,C.components):assert E==C.dummy_component or opts.same_type(D,opts.data_labels[B].default),f"Bad value for setting {B}: {D}; expecting {type(opts.data_labels[B].default).__name__}"
		for(B,D,E)in zip(opts.data_labels.keys(),F,C.components):
			if E==C.dummy_component:continue
			if opts.set(B,D):A.append(B)
		try:opts.save(shared.config_filename)
		except RuntimeError:return opts.dumpjson(),f"{len(A)} settings changed without save: {', '.join(A)}."
		return opts.dumpjson(),f"{len(A)} settings changed{': 'if A else''}{', '.join(A)}."
	def run_settings_single(C,value,key):
		B=value;A=key
		if not opts.same_type(B,opts.data_labels[A].default):return gr.update(visible=True),opts.dumpjson()
		if B is _A or not opts.set(A,B):return gr.update(value=getattr(opts,A)),opts.dumpjson()
		opts.save(shared.config_filename);return get_value_for_setting(A),opts.dumpjson()
	def create_ui(A,loadsave,dummy_component):
		G='licenses';H='download_localization';I='primary';F=dummy_component;A.components=[];A.component_dict={};A.dummy_component=F;shared.settings_components=A.component_dict;script_callbacks.ui_settings_callback();opts.reorder()
		with gr.Blocks(analytics_enabled=_B)as N:
			with gr.Row():
				with gr.Column(scale=6):A.submit=gr.Button(value='Apply settings',variant=I,elem_id='settings_submit')
				with gr.Column():O=gr.Button(value='Reload UI',variant=I,elem_id='settings_restart_gradio')
			A.result=gr.HTML(elem_id='settings_result');A.quicksettings_names=opts.quicksettings_list;A.quicksettings_names={A:B for(B,A)in enumerate(A.quicksettings_names)if A!=_C};A.quicksettings_list=[];J=_A;B=_A;D=_A
			with gr.Tabs(elem_id='settings'):
				for(P,(E,C))in enumerate(opts.data_labels.items()):
					K=C.section[0]is _A
					if J!=C.section and not K:
						Q,R=C.section
						if B is not _A:D.__exit__();B.__exit__()
						gr.Group();B=gr.TabItem(elem_id=f"settings_{Q}",label=R);B.__enter__();D=gr.Column(variant=_D);D.__enter__();J=C.section
					if E in A.quicksettings_names and not shared.cmd_opts.freeze_settings:A.quicksettings_list.append((P,E,C));A.components.append(F)
					elif K:A.components.append(F)
					else:L=create_setting_component(E);A.component_dict[E]=L;A.components.append(L)
				if B is not _A:D.__exit__();B.__exit__()
				with gr.TabItem('Defaults',id='defaults',elem_id='settings_tab_defaults'):loadsave.create_ui()
				with gr.TabItem('Sysinfo',id='sysinfo',elem_id='settings_tab_sysinfo'):
					gr.HTML('<a href="./internal/sysinfo-download" class="sysinfo_big_link" download>Download system info</a><br /><a href="./internal/sysinfo" target="_blank">(or open as text in a new page)</a>',elem_id='sysinfo_download')
					with gr.Row():
						with gr.Column(scale=1):M=gr.File(label='Check system info for validity',type='binary')
						with gr.Column(scale=1):S=gr.HTML('',elem_id='sysinfo_validity')
						with gr.Column(scale=100):0
				with gr.TabItem('Actions',id='actions',elem_id='settings_tab_actions'):
					T=gr.Button(value='Request browser notifications',elem_id='request_notifications');U=gr.Button(value='Download localization template',elem_id=H);V=gr.Button(value='Reload custom script bodies (No ui updates, No restart)',variant='secondary',elem_id='settings_reload_script_bodies')
					with gr.Row():W=gr.Button(value='Unload SD checkpoint to free VRAM',elem_id='sett_unload_sd_model');X=gr.Button(value='Reload the last SD checkpoint back into VRAM',elem_id='sett_reload_sd_model')
				with gr.TabItem('Licenses',id=G,elem_id='settings_tab_licenses'):gr.HTML(shared.html('licenses.html'),elem_id=G)
				gr.Button(value='Show all pages',elem_id='settings_show_all_pages');A.text_settings=gr.Textbox(elem_id='settings_json',value=lambda:opts.dumpjson(),visible=_B)
			W.click(fn=sd_models.unload_model_weights,inputs=[],outputs=[]);X.click(fn=sd_models.reload_model_weights,inputs=[],outputs=[]);T.click(fn=lambda:_A,inputs=[],outputs=[],_js='function(){}');U.click(fn=lambda:_A,inputs=[],outputs=[],_js=H)
			def Y():scripts.reload_script_body_only();reload_javascript()
			V.click(fn=Y,inputs=[],outputs=[]);O.click(fn=shared.state.request_restart,_js='restart_reload',inputs=[],outputs=[])
			def Z(x):
				if x is _A:return''
				if sysinfo.check(x.decode('utf8',errors='ignore')):return'Valid'
				return'Invalid'
			M.change(fn=Z,inputs=[M],outputs=[S])
		A.interface=N
	def add_quicksettings(A):
		with gr.Row(elem_id=_C,variant=_D):
			for(D,B,E)in sorted(A.quicksettings_list,key=lambda x:A.quicksettings_names.get(x[1],x[0])):C=create_setting_component(B,is_quicksettings=True);A.component_dict[B]=C
	def add_functionality(A,demo):
		C='sd_model_checkpoint';A.submit.click(fn=wrap_gradio_call(lambda*B:A.run_settings(*B),extra_outputs=[gr.update()]),inputs=A.components,outputs=[A.text_settings,A.result])
		for(K,D,L)in A.quicksettings_list:
			B=A.component_dict[D];G=opts.data_labels[D]
			if isinstance(B,gr.Textbox):E=[B.submit,B.blur]
			elif hasattr(B,'release'):E=[B.release]
			else:E=[B.change]
			for H in E:H(fn=lambda value,k=D:A.run_settings_single(value,key=k),inputs=[B],outputs=[B,A.text_settings],show_progress=G.refresh is not _A)
		I=gr.Button('Change checkpoint',elem_id='change_checkpoint',visible=_B);I.click(fn=lambda value,_:A.run_settings_single(value,key=C),_js="function(v){ var res = desiredCheckpointName; desiredCheckpointName = ''; return [res || v, null]; }",inputs=[A.component_dict[C],A.dummy_component],outputs=[A.component_dict[C],A.text_settings]);F=[B for B in opts.data_labels.keys()if B in A.component_dict]
		def J():return[get_value_for_setting(A)for A in F]
		demo.load(fn=J,inputs=[],outputs=[A.component_dict[B]for B in F],queue=_B)