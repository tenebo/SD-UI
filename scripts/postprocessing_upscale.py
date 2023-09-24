_E='Postprocess upscaler'
_D='upscale_by'
_C=False
_B='None'
_A=None
from PIL import Image
import numpy as np
from modules import scripts_postprocessing,shared
import gradio as gr
from modules.ui_components import FormRow,ToolButton
from modules.ui import switch_values_symbol
upscale_cache={}
class ScriptPostprocessingUpscale(scripts_postprocessing.ScriptPostprocessing):
	name='Upscale';order=1000
	def ui(L):
		A=gr.State(value=0)
		with gr.Column():
			with FormRow():
				with gr.Tabs(elem_id='extras_resize_mode'):
					with gr.TabItem('Scale by',elem_id='extras_scale_by_tab')as D:E=gr.Slider(minimum=1.,maximum=8.,step=.05,label='Resize',value=4,elem_id='extras_upscaling_resize')
					with gr.TabItem('Scale to',elem_id='extras_scale_to_tab')as F:
						with FormRow():
							with gr.Column(elem_id='upscaling_column_size',scale=4):B=gr.Slider(minimum=64,maximum=2048,step=8,label='Width',value=512,elem_id='extras_upscaling_resize_w');C=gr.Slider(minimum=64,maximum=2048,step=8,label='Height',value=512,elem_id='extras_upscaling_resize_h')
							with gr.Column(elem_id='upscaling_dimensions_row',scale=1,elem_classes='dimensions-tools'):G=ToolButton(value=switch_values_symbol,elem_id='upscaling_res_switch_btn');H=gr.Checkbox(label='Crop to fit',value=True,elem_id='extras_upscaling_crop')
			with FormRow():I=gr.Dropdown(label='Upscaler 1',elem_id='extras_upscaler_1',choices=[A.name for A in shared.sd_upscalers],value=shared.sd_upscalers[0].name)
			with FormRow():J=gr.Dropdown(label='Upscaler 2',elem_id='extras_upscaler_2',choices=[A.name for A in shared.sd_upscalers],value=shared.sd_upscalers[0].name);K=gr.Slider(minimum=.0,maximum=1.,step=.001,label='Upscaler 2 visibility',value=.0,elem_id='extras_upscaler_2_visibility')
		G.click(lambda w,h:(h,w),inputs=[B,C],outputs=[B,C],show_progress=_C);D.select(fn=lambda:0,inputs=[],outputs=[A]);F.select(fn=lambda:1,inputs=[],outputs=[A]);return{'upscale_mode':A,_D:E,'upscale_to_width':B,'upscale_to_height':C,'upscale_crop':H,'upscaler_1_name':I,'upscaler_2_name':J,'upscaler_2_visibility':K}
	def upscale(L,image,info,upscaler,upscale_mode,upscale_by,upscale_to_width,upscale_to_height,upscale_crop):
		H=upscale_crop;G=upscale_mode;F=upscaler;E=info;D=upscale_by;C=upscale_to_height;B=upscale_to_width;A=image
		if G==1:D=max(B/A.width,C/A.height);E['Postprocess upscale to']=f"{B}x{C}"
		else:E['Postprocess upscale by']=D
		I=hash(np.array(A.getdata()).tobytes()),F.name,G,D,B,C,H;J=upscale_cache.pop(I,_A)
		if J is not _A:A=J
		else:A=F.scaler.upscale(A,D,F.data_path)
		upscale_cache[I]=A
		if len(upscale_cache)>shared.opts.upscaling_max_images_in_cache:upscale_cache.pop(next(iter(upscale_cache),_A),_A)
		if G==1 and H:K=Image.new('RGB',(B,C));K.paste(A,box=(B//2-A.width//2,C//2-A.height//2));A=K;E['Postprocess crop to']=f"{A.width}x{A.height}"
		return A
	def process(G,pp,upscale_mode=1,upscale_by=2.,upscale_to_width=_A,upscale_to_height=_A,upscale_crop=_C,upscaler_1_name=_A,upscaler_2_name=_A,upscaler_2_visibility=.0):
		M=upscaler_2_visibility;L=upscale_crop;K=upscale_to_height;J=upscale_to_width;I=upscale_by;H=upscale_mode;C=upscaler_2_name;B=upscaler_1_name;A=pp
		if B==_B:B=_A
		D=next(iter([A for A in shared.sd_upscalers if A.name==B]),_A);assert D or B is _A,f"could not find upscaler named {B}"
		if not D:return
		if C==_B:C=_A
		E=next(iter([A for A in shared.sd_upscalers if A.name==C and A.name!=_B]),_A);assert E or C is _A,f"could not find upscaler named {C}";F=G.upscale(A.image,A.info,D,H,I,J,K,L);A.info[_E]=D.name
		if E and M>0:N=G.upscale(A.image,A.info,E,H,I,J,K,L);F=Image.blend(F,N,M);A.info['Postprocess upscaler 2']=E.name
		A.image=F
	def image_changed(A):upscale_cache.clear()
class ScriptPostprocessingUpscaleSimple(ScriptPostprocessingUpscale):
	name='Simple Upscale';order=900
	def ui(C):
		with FormRow():A=gr.Dropdown(label='Upscaler',choices=[A.name for A in shared.sd_upscalers],value=shared.sd_upscalers[0].name);B=gr.Slider(minimum=.05,maximum=8.,step=.05,label='Upscale by',value=2)
		return{_D:B,'upscaler_name':A}
	def process(C,pp,upscale_by=2.,upscaler_name=_A):
		A=upscaler_name
		if A is _A or A==_B:return
		B=next(iter([B for B in shared.sd_upscalers if B.name==A]),_A);assert B,f"could not find upscaler named {A}";pp.image=C.upscale(pp.image,pp.info,B,0,upscale_by,0,0,_C);pp.info[_E]=B.name