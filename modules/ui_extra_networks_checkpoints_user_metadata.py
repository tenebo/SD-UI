import gradio as gr
from modules import ui_extra_networks_user_metadata,sd_vae,shared
from modules.ui_common import create_refresh_button
class CheckpointUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
	def __init__(A,ui,tabname,page):super().__init__(ui,tabname,page);A.select_vae=None
	def save_user_metadata(B,name,desc,notes,vae):A=B.get_user_metadata(name);A['description']=desc;A['notes']=notes;A['vae']=vae;B.write_user_metadata(name,A)
	def update_vae(A,name):
		if name==shared.sd_model.sd_checkpoint_info.name_for_extra:sd_vae.reload_vae_weights()
	def put_values_into_components(A,name):B=A.get_user_metadata(name);C=super().put_values_into_components(name);return[*C[0:5],B.get('vae','')]
	def create_editor(A):
		C='Automatic';B='None';A.create_default_editor_elems()
		with gr.Row():A.select_vae=gr.Dropdown(choices=[C,B]+list(sd_vae.vae_dict),value=B,label='Preferred VAE',elem_id='checpoint_edit_user_metadata_preferred_vae');create_refresh_button(A.select_vae,sd_vae.refresh_vae_list,lambda:{'choices':[C,B]+list(sd_vae.vae_dict)},'checpoint_edit_user_metadata_refresh_preferred_vae')
		A.edit_notes=gr.TextArea(label='Notes',lines=4);A.create_default_buttons();D=[A.edit_name,A.edit_description,A.html_filedata,A.html_preview,A.edit_notes,A.select_vae];A.button_edit.click(fn=A.put_values_into_components,inputs=[A.edit_name_input],outputs=D).then(fn=lambda:gr.update(visible=True),inputs=[],outputs=[A.box]);E=[A.edit_description,A.edit_notes,A.select_vae];A.setup_save_handler(A.button_save,A.save_user_metadata,E);A.button_save.click(fn=A.update_vae,inputs=[A.edit_name_input])