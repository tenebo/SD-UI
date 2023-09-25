_A='GFPGAN visibility'
from PIL import Image
import numpy as np
from modules import scripts_postprocessing,gfpgan_model
import gradio as gr
from modules.ui_components import FormRow
class ScriptPostprocessingGfpGan(scripts_postprocessing.ScriptPostprocessing):
	name='GFPGAN';order=2000
	def ui(B):
		with FormRow():A=gr.Slider(minimum=.0,maximum=1.,step=.001,label=_A,value=0,elem_id='extras_gfpgan_visibility')
		return{'gfpgan_visibility':A}
	def process(D,pp,gfpgan_visibility):
		A=gfpgan_visibility
		if A==0:return
		C=gfpgan_model.gfpgan_fix_faces(np.array(pp.image,dtype=np.uint8));B=Image.fromarray(C)
		if A<1.:B=Image.blend(pp.image,B,A)
		pp.image=B;pp.info[_A]=round(A,3)