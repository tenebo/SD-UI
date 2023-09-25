_B=False
_A=None
import json,os,gradio as gr
from modules import errors
from modules.ui_components import ToolButton
def radio_choices(comp):return[A[0]if isinstance(A,tuple)else A for A in getattr(comp,'choices',[])]
class UiLoadsave:
	'allows saving and restoring default values for gradio components'
	def __init__(A,filename):
		A.filename=filename;A.ui_settings={};A.component_mapping={};A.error_loading=_B;A.finalized_ui=_B;A.ui_defaults_view=_A;A.ui_defaults_apply=_A;A.ui_defaults_review=_A
		try:
			if os.path.exists(A.filename):A.ui_settings=A.read_from_file()
		except Exception as B:A.error_loading=True;errors.display(B,'loading settings')
	def add_component(E,path,x):
		'adds component to the registry of tracked components';B='value';assert not E.finalized_ui
		def A(obj,field,condition=_A,init_field=_A):
			H=init_field;G=condition;F=obj;C=field;D=f"{path}/{C}"
			if getattr(F,'custom_script_source',_A)is not _A:D=f"customscript/{F.custom_script_source}/{D}"
			if getattr(F,'do_not_save_to_config',_B):return
			A=E.ui_settings.get(D,_A)
			if A is _A:E.ui_settings[D]=getattr(F,C)
			elif G and not G(A):0
			else:
				if isinstance(x,gr.Textbox)and C==B:A=str(A)
				elif isinstance(x,gr.Number)and C==B:
					try:A=float(A)
					except ValueError:return
				setattr(F,C,A)
				if H is not _A:H(A)
			if C==B and D not in E.component_mapping:E.component_mapping[D]=x
		if type(x)in[gr.Slider,gr.Radio,gr.Checkbox,gr.Textbox,gr.Number,gr.Dropdown,ToolButton,gr.Button]and x.visible:A(x,'visible')
		if type(x)==gr.Slider:A(x,B);A(x,'minimum');A(x,'maximum');A(x,'step')
		if type(x)==gr.Radio:A(x,B,lambda val:val in radio_choices(x))
		if type(x)==gr.Checkbox:A(x,B)
		if type(x)==gr.Textbox:A(x,B)
		if type(x)==gr.Number:A(x,B)
		if type(x)==gr.Dropdown:
			def C(val):
				A=radio_choices(x)
				if getattr(x,'multiselect',_B):return all(B in A for B in val)
				else:return val in A
			A(x,B,C,getattr(x,'init_field',_A))
		def D(tab_id):
			A=tab_id;B=list(filter(lambda e:isinstance(e,gr.TabItem),x.children))
			if type(A)==str:C=[A.id for A in B];return A in C
			elif type(A)==int:return 0<=A<len(B)
			else:return _B
		if type(x)==gr.Tabs:A(x,'selected',D)
	def add_block(A,x,path=''):
		'adds all components inside a gradio block x to the registry of tracked components';B=path
		if hasattr(x,'children'):
			if isinstance(x,gr.Tabs)and x.elem_id is not _A:A.add_component(f"{B}/Tabs@{x.elem_id}",x)
			for C in x.children:A.add_block(C,B)
		elif x.label is not _A:A.add_component(f"{B}/{x.label}",x)
		elif isinstance(x,gr.Button)and x.value is not _A:A.add_component(f"{B}/{x.value}",x)
	def read_from_file(A):
		with open(A.filename,'r',encoding='utf8')as B:return json.load(B)
	def write_to_file(A,current_ui_settings):
		with open(A.filename,'w',encoding='utf8')as B:json.dump(current_ui_settings,B,indent=4)
	def dump_defaults(A):
		'saves default values to a file unless tjhe file is present and there was an error loading default values at start'
		if A.error_loading and os.path.exists(A.filename):return
		A.write_to_file(A.ui_settings)
	def iter_changes(E,current_ui_settings,values):
		'\n        given a dictionary with defaults from a file and current values from gradio elements, returns\n        an iterator over tuples of values that are not the same between the file and the current;\n        tuple contents are: path, old value, new value\n        '
		for((D,F),A)in zip(E.component_mapping.items(),values):
			B=current_ui_settings.get(D);C=radio_choices(F)
			if isinstance(A,int)and C:
				if A>=len(C):continue
				A=C[A]
				if isinstance(A,tuple):A=A[0]
			if A==B:continue
			if B is _A and A==''or A==[]:continue
			yield(D,B,A)
	def ui_view(C,*D):
		A=['<table><thead><tr><th>Path</th><th>Old value</th><th>New value</th></thead><tbody>']
		for(E,B,F)in C.iter_changes(C.read_from_file(),D):
			if B is _A:B="<span class='ui-defaults-none'>None</span>"
			A.append(f"<tr><td>{E}</td><td>{B}</td><td>{F}</td></tr>")
		if len(A)==1:A.append('<tr><td colspan=3>No changes</td></tr>')
		A.append('</tbody>');return''.join(A)
	def ui_apply(A,*D):
		B=0;C=A.read_from_file()
		for(E,G,F)in A.iter_changes(C.copy(),D):B+=1;C[E]=F
		if B==0:return'No changes.'
		A.write_to_file(C);return f"Wrote {B} changes."
	def create_ui(A):
		'creates ui elements for editing defaults UI, without adding any logic to them';gr.HTML(f"This page allows you to change default values in UI elements on other tabs.<br />Make your changes, press 'View changes' to review the changed default values,<br />then press 'Apply' to write them to {A.filename}.<br />New defaults will apply after you restart the UI.<br />")
		with gr.Row():A.ui_defaults_view=gr.Button(value='View changes',elem_id='ui_defaults_view',variant='secondary');A.ui_defaults_apply=gr.Button(value='Apply',elem_id='ui_defaults_apply',variant='primary')
		A.ui_defaults_review=gr.HTML('')
	def setup_ui(A):'adds logic to elements created with create_ui; all add_block class must be made before this';assert not A.finalized_ui;A.finalized_ui=True;A.ui_defaults_view.click(fn=A.ui_view,inputs=list(A.component_mapping.values()),outputs=[A.ui_defaults_review]);A.ui_defaults_apply.click(fn=A.ui_apply,inputs=list(A.component_mapping.values()),outputs=[A.ui_defaults_review])