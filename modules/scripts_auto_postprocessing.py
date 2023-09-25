from modules import scripts,scripts_postprocessing,shared
class ScriptPostprocessingForMainUI(scripts.Script):
	def __init__(A,script_postproc):A.script=script_postproc;A.postprocessing_controls=None
	def title(A):return A.script.name
	def show(A,is_img2img):return scripts.AlwaysVisible
	def ui(A,is_img2img):A.postprocessing_controls=A.script.ui();return A.postprocessing_controls.values()
	def postprocess_image(B,p,script_pp,*D):C=script_pp;E=dict(zip(B.postprocessing_controls,D));A=scripts_postprocessing.PostprocessedImage(C.image);A.info={};B.script.process(A,**E);p.extra_generation_params.update(A.info);C.image=A.image
def create_auto_preprocessing_script_data():
	from modules import scripts as B;C=[]
	for D in shared.opts.postprocessing_enable_in_main_ui:
		A=next(iter([A for A in B.postprocessing_scripts_data if A.script_class.name==D]),None)
		if A is None:continue
		E=lambda s=A:ScriptPostprocessingForMainUI(s.script_class());C.append(B.ScriptClassData(script_class=E,path=A.path,basedir=A.basedir,module=A.module))
	return C