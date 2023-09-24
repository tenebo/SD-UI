_F='Add difference'
_E='No interpolation'
_D='safetensors'
_C=True
_B='Weighted sum'
_A=False
import gradio as gr
from modules import sd_models,sd_vae,errors,extras,call_queue
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button
def update_interp_description(value):A="<p style='margin-bottom: 2.5em'>{}</p>";B={_E:A.format('No interpolation will be used. Requires one model; A. Allows for format conversion and VAE baking.'),_B:A.format('A weighted sum will be used for interpolation. Requires two models; A and B. The result is calculated as A * (1 - M) + B * M'),_F:A.format('The difference between the last two models will be added to the first. Requires three models; A, B and C. The result is calculated as A + (B - C) * M')};return B[value]
def modelmerger(*A):
	try:B=extras.run_modelmerger(*A)
	except Exception as C:errors.report('Error loading/saving model file',exc_info=_C);sd_models.list_models();return[*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles())for A in range(4)],f"Error merging checkpoints: {C}"]
	return B
class UiCheckpointMerger:
	def __init__(A):
		D='A, B or C';E='compact';C='None';B='choices'
		with gr.Blocks(analytics_enabled=_A)as F:
			with gr.Row(equal_height=_A):
				with gr.Column(variant=E):
					A.interp_description=gr.HTML(value=update_interp_description(_B),elem_id='modelmerger_interp_description')
					with FormRow(elem_id='modelmerger_models'):A.primary_model_name=gr.Dropdown(sd_models.checkpoint_tiles(),elem_id='modelmerger_primary_model_name',label='Primary model (A)');create_refresh_button(A.primary_model_name,sd_models.list_models,lambda:{B:sd_models.checkpoint_tiles()},'refresh_checkpoint_A');A.secondary_model_name=gr.Dropdown(sd_models.checkpoint_tiles(),elem_id='modelmerger_secondary_model_name',label='Secondary model (B)');create_refresh_button(A.secondary_model_name,sd_models.list_models,lambda:{B:sd_models.checkpoint_tiles()},'refresh_checkpoint_B');A.tertiary_model_name=gr.Dropdown(sd_models.checkpoint_tiles(),elem_id='modelmerger_tertiary_model_name',label='Tertiary model (C)');create_refresh_button(A.tertiary_model_name,sd_models.list_models,lambda:{B:sd_models.checkpoint_tiles()},'refresh_checkpoint_C')
					A.custom_name=gr.Textbox(label='Custom Name (Optional)',elem_id='modelmerger_custom_name');A.interp_amount=gr.Slider(minimum=.0,maximum=1.,step=.05,label='Multiplier (M) - set to 0 to get model A',value=.3,elem_id='modelmerger_interp_amount');A.interp_method=gr.Radio(choices=[_E,_B,_F],value=_B,label='Interpolation Method',elem_id='modelmerger_interp_method');A.interp_method.change(fn=update_interp_description,inputs=[A.interp_method],outputs=[A.interp_description])
					with FormRow():A.checkpoint_format=gr.Radio(choices=['ckpt',_D],value=_D,label='Checkpoint format',elem_id='modelmerger_checkpoint_format');A.save_as_half=gr.Checkbox(value=_A,label='Save as float16',elem_id='modelmerger_save_as_half')
					with FormRow():
						with gr.Column():A.config_source=gr.Radio(choices=[D,'B','C',"Don't"],value=D,label='Copy config from',type='index',elem_id='modelmerger_config_method')
						with gr.Column():
							with FormRow():A.bake_in_vae=gr.Dropdown(choices=[C]+list(sd_vae.vae_dict),value=C,label='Bake in VAE',elem_id='modelmerger_bake_in_vae');create_refresh_button(A.bake_in_vae,sd_vae.refresh_vae_list,lambda:{B:[C]+list(sd_vae.vae_dict)},'modelmerger_refresh_bake_in_vae')
					with FormRow():A.discard_weights=gr.Textbox(value='',label='Discard weights with matching name',elem_id='modelmerger_discard_weights')
					with gr.Accordion('Metadata',open=_A)as G:
						with FormRow():A.save_metadata=gr.Checkbox(value=_C,label='Save metadata',elem_id='modelmerger_save_metadata');A.add_merge_recipe=gr.Checkbox(value=_C,label='Add merge recipe metadata',elem_id='modelmerger_add_recipe');A.copy_metadata_fields=gr.Checkbox(value=_C,label='Copy metadata from merged models',elem_id='modelmerger_copy_metadata')
						A.metadata_json=gr.TextArea('{}',label='Metadata in JSON format');A.read_metadata=gr.Button('Read metadata from selected checkpoints')
					with FormRow():A.modelmerger_merge=gr.Button(elem_id='modelmerger_merge',value='Merge',variant='primary')
				with gr.Column(variant=E,elem_id='modelmerger_results_container'):
					with gr.Group(elem_id='modelmerger_results_panel'):A.modelmerger_result=gr.HTML(elem_id='modelmerger_result',show_label=_A)
		A.metadata_editor=G;A.blocks=F
	def setup_ui(A,dummy_component,sd_model_checkpoint_component):A.checkpoint_format.change(lambda fmt:gr.update(visible=fmt==_D),inputs=[A.checkpoint_format],outputs=[A.metadata_editor],show_progress=_A);A.read_metadata.click(extras.read_metadata,inputs=[A.primary_model_name,A.secondary_model_name,A.tertiary_model_name],outputs=[A.metadata_json]);A.modelmerger_merge.click(fn=lambda:'',inputs=[],outputs=[A.modelmerger_result]);A.modelmerger_merge.click(fn=call_queue.wrap_gradio_gpu_call(modelmerger,extra_outputs=lambda:[gr.update()for A in range(4)]),_js='modelmerger',inputs=[dummy_component,A.primary_model_name,A.secondary_model_name,A.tertiary_model_name,A.interp_method,A.interp_amount,A.save_as_half,A.custom_name,A.checkpoint_format,A.config_source,A.bake_in_vae,A.discard_weights,A.save_metadata,A.add_merge_recipe,A.copy_metadata_fields,A.metadata_json],outputs=[A.primary_model_name,A.secondary_model_name,A.tertiary_model_name,sd_model_checkpoint_component,A.modelmerger_result]);A.interp_description.value=update_interp_description(A.interp_method.value)