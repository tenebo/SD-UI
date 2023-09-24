import gradio as gr
from modules import scripts,shared,ui_common,postprocessing,call_queue
import modules.generation_parameters_copypaste as parameters_copypaste
def create_ui():
	F='extras';E='Batch Process';D='compact';C=True;A=gr.State(value=0)
	with gr.Row(equal_height=False,variant=D):
		with gr.Column(variant=D):
			with gr.Tabs(elem_id='mode_extras'):
				with gr.TabItem('Single Image',id='single_image',elem_id='extras_single_tab')as G:B=gr.Image(label='Source',source='upload',interactive=C,type='pil',elem_id='extras_image')
				with gr.TabItem(E,id='batch_process',elem_id='extras_batch_process_tab')as H:I=gr.Files(label=E,interactive=C,elem_id='extras_image_batch')
				with gr.TabItem('Batch from Directory',id='batch_from_directory',elem_id='extras_batch_directory_tab')as J:K=gr.Textbox(label='Input directory',**shared.hide_dirs,placeholder='A directory on the same machine where the server is running.',elem_id='extras_batch_input_dir');L=gr.Textbox(label='Output directory',**shared.hide_dirs,placeholder='Leave blank to save images to the default path.',elem_id='extras_batch_output_dir');M=gr.Checkbox(label='Show result images',value=C,elem_id='extras_show_extras_results')
			N=gr.Button('Generate',elem_id='extras_generate',variant='primary');O=scripts.scripts_postproc.setup_ui()
		with gr.Column():P,Q,R,S=ui_common.create_output_panel(F,shared.opts.outdir_extras_samples)
	G.select(fn=lambda:0,inputs=[],outputs=[A]);H.select(fn=lambda:1,inputs=[],outputs=[A]);J.select(fn=lambda:2,inputs=[],outputs=[A]);N.click(fn=call_queue.wrap_gradio_gpu_call(postprocessing.run_postprocessing,extra_outputs=[None,'']),inputs=[A,B,I,K,L,M,*O],outputs=[P,Q,R]);parameters_copypaste.add_paste_fields(F,B,None);B.change(fn=scripts.scripts_postproc.image_changed,inputs=[],outputs=[])