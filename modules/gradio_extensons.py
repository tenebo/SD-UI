_C='__init__'
_B='example_inputs'
_A=None
import gradio as gr
from modules import scripts,ui_tempdir,patches
def add_classes_to_gradio_component(comp):
	'\n    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others\n    ';B='multiselect';A=comp;A.elem_classes=[f"gradio-{A.get_block_name()}",*(A.elem_classes or[])]
	if getattr(A,B,False):A.elem_classes.append(B)
def IOComponent_init(self,*C,**B):
	A=self;A.webui_tooltip=B.pop('tooltip',_A)
	if scripts.scripts_current is not _A:scripts.scripts_current.before_component(A,**B)
	scripts.script_callbacks.before_component_callback(A,**B);D=original_IOComponent_init(A,*C,**B);add_classes_to_gradio_component(A);scripts.script_callbacks.after_component_callback(A,**B)
	if scripts.scripts_current is not _A:scripts.scripts_current.after_component(A,**B)
	return D
def Block_get_config(self):
	C='webui_tooltip';A=original_Block_get_config(self);B=getattr(self,C,_A)
	if B:A[C]=B
	A.pop(_B,_A);return A
def BlockContext_init(self,*A,**B):C=original_BlockContext_init(self,*A,**B);add_classes_to_gradio_component(self);return C
def Blocks_get_config_file(self,*C,**D):
	A=original_Blocks_get_config_file(self,*C,**D)
	for B in A['components']:
		if _B in B:B[_B]={'serialized':[]}
	return A
original_IOComponent_init=patches.patch(__name__,obj=gr.components.IOComponent,field=_C,replacement=IOComponent_init)
original_Block_get_config=patches.patch(__name__,obj=gr.blocks.Block,field='get_config',replacement=Block_get_config)
original_BlockContext_init=patches.patch(__name__,obj=gr.blocks.BlockContext,field=_C,replacement=BlockContext_init)
original_Blocks_get_config_file=patches.patch(__name__,obj=gr.blocks.Blocks,field='get_config_file',replacement=Blocks_get_config_file)
ui_tempdir.install_ui_tempdir_override()