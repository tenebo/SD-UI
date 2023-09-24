_B=True
_A=False
import gradio as gr
from modules import shared,ui_common,ui_components,styles
styles_edit_symbol='🖌️'
styles_materialize_symbol='📋'
def select_style(name):A=shared.prompt_styles.styles.get(name);B=A is not None;C=not name;D=A.prompt if A else gr.update();E=A.negative_prompt if A else gr.update();return D,E,gr.update(visible=B),gr.update(visible=not C)
def save_style(name,prompt,negative_prompt):
	if not name:return gr.update(visible=_A)
	A=styles.PromptStyle(name,prompt,negative_prompt);shared.prompt_styles.styles[A.name]=A;shared.prompt_styles.save_styles(shared.styles_filename);return gr.update(visible=_B)
def delete_style(name):
	if name=='':return
	shared.prompt_styles.styles.pop(name,None);shared.prompt_styles.save_styles(shared.styles_filename);return'','',''
def materialize_styles(prompt,negative_prompt,styles):C=styles;A=negative_prompt;B=prompt;B=shared.prompt_styles.apply_styles_to_prompt(B,C);A=shared.prompt_styles.apply_negative_styles_to_prompt(A,C);return[gr.Textbox.update(value=B),gr.Textbox.update(value=A),gr.Dropdown.update(value=[])]
def refresh_styles():return gr.update(choices=list(shared.prompt_styles.styles)),gr.update(choices=list(shared.prompt_styles.styles))
class UiPromptStyles:
	def __init__(A,tabname,main_ui_prompt,main_ui_negative_prompt):
		D='primary';E=main_ui_negative_prompt;F=main_ui_prompt;C='Styles';B=tabname;A.tabname=B
		with gr.Row(elem_id=f"{B}_styles_row"):A.dropdown=gr.Dropdown(label=C,show_label=_A,elem_id=f"{B}_styles",choices=list(shared.prompt_styles.styles),value=[],multiselect=_B,tooltip=C);G=ui_components.ToolButton(value=styles_edit_symbol,elem_id=f"{B}_styles_edit_button",tooltip='Edit styles')
		with gr.Box(elem_id=f"{B}_styles_dialog",elem_classes='popup-dialog')as H:
			with gr.Row():A.selection=gr.Dropdown(label=C,elem_id=f"{B}_styles_edit_select",choices=list(shared.prompt_styles.styles),value=[],allow_custom_value=_B,info="Styles allow you to add custom text to prompt. Use the {prompt} token in style text, and it will be replaced with user's prompt when applying style. Otherwise, style's text will be added to the end of the prompt.");ui_common.create_refresh_button([A.dropdown,A.selection],shared.prompt_styles.reload,lambda:{'choices':list(shared.prompt_styles.styles)},f"refresh_{B}_styles");A.materialize=ui_components.ToolButton(value=styles_materialize_symbol,elem_id=f"{B}_style_apply",tooltip='Apply all selected styles from the style selction dropdown in main UI to the prompt.')
			with gr.Row():A.prompt=gr.Textbox(label='Prompt',show_label=_B,elem_id=f"{B}_edit_style_prompt",lines=3)
			with gr.Row():A.neg_prompt=gr.Textbox(label='Negative prompt',show_label=_B,elem_id=f"{B}_edit_style_neg_prompt",lines=3)
			with gr.Row():A.save=gr.Button('Save',variant=D,elem_id=f"{B}_edit_style_save",visible=_A);A.delete=gr.Button('Delete',variant=D,elem_id=f"{B}_edit_style_delete",visible=_A);A.close=gr.Button('Close',variant='secondary',elem_id=f"{B}_edit_style_close")
		A.selection.change(fn=select_style,inputs=[A.selection],outputs=[A.prompt,A.neg_prompt,A.delete,A.save],show_progress=_A);A.save.click(fn=save_style,inputs=[A.selection,A.prompt,A.neg_prompt],outputs=[A.delete],show_progress=_A).then(refresh_styles,outputs=[A.dropdown,A.selection],show_progress=_A);A.delete.click(fn=delete_style,_js='function(name){ if(name == "") return ""; return confirm("Delete style " + name + "?") ? name : ""; }',inputs=[A.selection],outputs=[A.selection,A.prompt,A.neg_prompt],show_progress=_A).then(refresh_styles,outputs=[A.dropdown,A.selection],show_progress=_A);A.materialize.click(fn=materialize_styles,inputs=[F,E,A.dropdown],outputs=[F,E,A.dropdown],show_progress=_A).then(fn=None,_js='function(){update_'+B+'_tokens(); closePopup();}',show_progress=_A);ui_common.setup_dialog(button_show=G,dialog=H,button_close=A.close)