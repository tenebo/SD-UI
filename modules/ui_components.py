_B='dropdown'
_A='elem_classes'
import gradio as gr
class FormComponent:
	def get_expected_parent(A):return gr.components.Form
gr.Dropdown.get_expected_parent=FormComponent.get_expected_parent
class ToolButton(FormComponent,gr.Button):
	'Small button with single emoji as text, fits inside gradio forms'
	def __init__(D,*B,**A):C=A.pop(_A,[]);super().__init__(*B,elem_classes=['tool',*C],**A)
	def get_block_name(A):return'button'
class ResizeHandleRow(gr.Row):
	'Same as gr.Row but fits inside gradio forms'
	def __init__(A,**B):super().__init__(**B);A.elem_classes.append('resize-handle-row')
	def get_block_name(A):return'row'
class FormRow(FormComponent,gr.Row):
	'Same as gr.Row but fits inside gradio forms'
	def get_block_name(A):return'row'
class FormColumn(FormComponent,gr.Column):
	'Same as gr.Column but fits inside gradio forms'
	def get_block_name(A):return'column'
class FormGroup(FormComponent,gr.Group):
	'Same as gr.Group but fits inside gradio forms'
	def get_block_name(A):return'group'
class FormHTML(FormComponent,gr.HTML):
	'Same as gr.HTML but fits inside gradio forms'
	def get_block_name(A):return'html'
class FormColorPicker(FormComponent,gr.ColorPicker):
	'Same as gr.ColorPicker but fits inside gradio forms'
	def get_block_name(A):return'colorpicker'
class DropdownMulti(FormComponent,gr.Dropdown):
	'Same as gr.Dropdown but always multiselect'
	def __init__(B,**A):super().__init__(multiselect=True,**A)
	def get_block_name(A):return _B
class DropdownEditable(FormComponent,gr.Dropdown):
	'Same as gr.Dropdown but allows editing value'
	def __init__(B,**A):super().__init__(allow_custom_value=True,**A)
	def get_block_name(A):return _B
class InputAccordion(gr.Checkbox):
	'A gr.Accordion that can be used as an input - returns True if open, False if closed.\n\n    Actaully just a hidden checkbox, but creates an accordion that follows and is followed by the state of the checkbox.\n    ';global_index=0
	def __init__(A,value,**B):
		E='label';D=value;C='elem_id';A.accordion_id=B.get(C)
		if A.accordion_id is None:A.accordion_id=f"input-accordion-{InputAccordion.global_index}";InputAccordion.global_index+=1
		F={**B,C:f"{A.accordion_id}-checkbox",'visible':False};super().__init__(D,**F);A.change(fn=None,_js='function(checked){ inputAccordionChecked("'+A.accordion_id+'", checked); }',inputs=[A]);G={**B,C:A.accordion_id,E:B.get(E,'Accordion'),_A:['input-accordion'],'open':D};A.accordion=gr.Accordion(**G)
	def extra(A):'Allows you to put something into the label of the accordion.\n\n        Use it like this:\n\n        ```\n        with InputAccordion(False, label="Accordion") as acc:\n            with acc.extra():\n                FormHTML(value="hello", min_width=0)\n\n            ...\n        ```\n        ';return gr.Column(elem_id=A.accordion_id+'-extra',elem_classes='input-accordion-extra',min_width=0)
	def __enter__(A):A.accordion.__enter__();return A
	def __exit__(A,exc_type,exc_val,exc_tb):A.accordion.__exit__(exc_type,exc_val,exc_tb)
	def get_block_name(A):return'checkbox'