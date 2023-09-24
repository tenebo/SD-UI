_G='basedir'
_F='title'
_E='txt2img'
_D='img2img'
_C=False
_B=True
_A=None
import os,re,sys,inspect
from collections import namedtuple
from dataclasses import dataclass
import gradio as gr
from modules import shared,paths,script_callbacks,extensions,script_loading,scripts_postprocessing,errors,timer
AlwaysVisible=object()
class PostprocessImageArgs:
	def __init__(A,image):A.image=image
class PostprocessBatchListArgs:
	def __init__(A,images):A.images=images
@dataclass
class OnComponent:component:gr.blocks.Block
class Script:
	name=_A;"script's internal name derived from title";section=_A;"name of UI section that the script's controls will be placed into";filename=_A;args_from=_A;args_to=_A;alwayson=_C;is_txt2img=_C;is_img2img=_C;tabname=_A;group=_A;"A gr.Group component that has all script's UI inside it.";create_group=_B;'If False, for alwayson scripts, a group component will not be created.';infotext_fields=_A;"if set in ui(), this is a list of pairs of gradio component + text; the text will be used when\n    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example\n    ";paste_field_names=_A;'if set in ui(), this is a list of names of infotext fields; the fields will be sent through the\n    various "Send to <X>" buttons when clicked\n    ';api_info=_A;'Generated value of type modules.api.models.ScriptInfo with information about the script for API';on_before_component_elem_id=_A;'list of callbacks to be called before a component with an elem_id is created';on_after_component_elem_id=_A;'list of callbacks to be called after a component with an elem_id is created';setup_for_ui_only=_C;'If true, the script setup will only be run in Gradio UI, not in API'
	def title(A):'this function should return the title of the script. This is what will be displayed in the dropdown menu.';raise NotImplementedError()
	def ui(A,is_img2img):'this function should create gradio UI elements. See https://gradio.app/docs/#components\n        The return value should be an array of all components that are used in processing.\n        Values of those returned components will be passed to run() and process() functions.\n        '
	def show(A,is_img2img):"\n        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise\n\n        This function should return:\n         - False if the script should not be shown in UI at all\n         - True if the script should be shown in UI if it's selected in the scripts dropdown\n         - script.AlwaysVisible if the script should be shown in UI at all times\n         ";return _B
	def run(A,p,*B):'\n        This function is called if the script has been selected in the script dropdown.\n        It must do all processing and return the Processed object with results, same as\n        one returned by processing.process_images.\n\n        Usually the processing is done by calling the processing.process_images function.\n\n        args contains all values returned by components from ui()\n        '
	def setup(A,p,*B):'For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.\n        args contains all values returned by components from ui().\n        '
	def before_process(A,p,*B):'\n        This function is called very early during processing begins for AlwaysVisible scripts.\n        You can modify the processing object (p) here, inject hooks, etc.\n        args contains all values returned by components from ui()\n        '
	def process(A,p,*B):'\n        This function is called before processing begins for AlwaysVisible scripts.\n        You can modify the processing object (p) here, inject hooks, etc.\n        args contains all values returned by components from ui()\n        '
	def before_process_batch(A,p,*B,**C):'\n        Called before extra networks are parsed from the prompt, so you can add\n        new extra network keywords to the prompt with this callback.\n\n        **kwargs will have those items:\n          - batch_number - index of current batch, from 0 to number of batches-1\n          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things\n          - seeds - list of seeds for current batch\n          - subseeds - list of subseeds for current batch\n        '
	def after_extra_networks_activate(A,p,*B,**C):"\n        Called after extra networks activation, before conds calculation\n        allow modification of the network after extra networks activation been applied\n        won't be call if p.disable_extra_networks\n\n        **kwargs will have those items:\n          - batch_number - index of current batch, from 0 to number of batches-1\n          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things\n          - seeds - list of seeds for current batch\n          - subseeds - list of subseeds for current batch\n          - extra_network_data - list of ExtraNetworkParams for current stage\n        "
	def process_batch(A,p,*B,**C):'\n        Same as process(), but called for every batch.\n\n        **kwargs will have those items:\n          - batch_number - index of current batch, from 0 to number of batches-1\n          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things\n          - seeds - list of seeds for current batch\n          - subseeds - list of subseeds for current batch\n        '
	def postprocess_batch(A,p,*B,**C):'\n        Same as process_batch(), but called for every batch after it has been generated.\n\n        **kwargs will have same items as process_batch, and also:\n          - batch_number - index of current batch, from 0 to number of batches-1\n          - images - torch tensor with all generated images, with values ranging from 0 to 1;\n        '
	def postprocess_batch_list(A,p,pp,*B,**C):'\n        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.\n        This is useful when you want to update the entire batch instead of individual images.\n\n        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.\n        If the number of images is different from the batch size when returning,\n        then the script has the responsibility to also update the following attributes in the processing object (p):\n          - p.prompts\n          - p.negative_prompts\n          - p.seeds\n          - p.subseeds\n\n        **kwargs will have same items as process_batch, and also:\n          - batch_number - index of current batch, from 0 to number of batches-1\n        '
	def postprocess_image(A,p,pp,*B):'\n        Called for every image after it has been generated.\n        '
	def postprocess(A,p,processed,*B):'\n        This function is called after processing ends for AlwaysVisible scripts.\n        args contains all values returned by components from ui()\n        '
	def before_component(A,component,**B):'\n        Called before a component is created.\n        Use elem_id/label fields of kwargs to figure out which component it is.\n        This can be useful to inject your own components somewhere in the middle of vanilla UI.\n        You can return created components in the ui() function to add them to the list of arguments for your processing functions\n        '
	def after_component(A,component,**B):'\n        Called after a component is created. Same as above.\n        '
	def on_before_component(A,callback,*,elem_id):
		"\n        Calls callback before a component is created. The callback function is called with a single argument of type OnComponent.\n\n        May be called in show() or ui() - but it may be too late in latter as some components may already be created.\n\n        This function is an alternative to before_component in that it also cllows to run before a component is created, but\n        it doesn't require to be called for every created component - just for the one you need.\n        "
		if A.on_before_component_elem_id is _A:A.on_before_component_elem_id=[]
		A.on_before_component_elem_id.append((elem_id,callback))
	def on_after_component(A,callback,*,elem_id):
		'\n        Calls callback after a component is created. The callback function is called with a single argument of type OnComponent.\n        '
		if A.on_after_component_elem_id is _A:A.on_after_component_elem_id=[]
		A.on_after_component_elem_id.append((elem_id,callback))
	def describe(A):'unused';return''
	def elem_id(A,item_id):'helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id';B=A.show(_B)==A.show(_C);C=_D if A.is_img2img else _E;D=f"{C}_"if B else'';E=re.sub('[^a-z_0-9]','',re.sub('\\s','_',A.title().lower()));return f"script_{D}{E}_{item_id}"
	def before_hr(A,p,*B):'\n        This function is called before hires fix start.\n        '
class ScriptBuiltinUI(Script):
	setup_for_ui_only=_B
	def elem_id(A,item_id):'helper function to generate id for a HTML element, constructs final id out of tab and user-supplied item_id';B=A.show(_B)==A.show(_C);C=(_D if A.is_img2img else _E)+'_'if B else'';return f"{C}{item_id}"
current_basedir=paths.script_path
def basedir():"returns the base directory for the current script. For scripts in the main scripts directory,\n    this is the main directory (where webui.py resides), and for scripts in extensions directory\n    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)\n    ";return current_basedir
ScriptFile=namedtuple('ScriptFile',[_G,'filename','path'])
scripts_data=[]
postprocessing_scripts_data=[]
ScriptClassData=namedtuple('ScriptClassData',['script_class','path',_G,'module'])
def list_scripts(scriptdirname,extension,*,include_extensions=_B):
	C=extension;D=scriptdirname;A=[];B=os.path.join(paths.script_path,D)
	if os.path.exists(B):
		for E in sorted(os.listdir(B)):A.append(ScriptFile(paths.script_path,E,os.path.join(B,E)))
	if include_extensions:
		for F in extensions.active():A+=F.list_files(D,C)
	A=[A for A in A if os.path.splitext(A.path)[1].lower()==C and os.path.isfile(A.path)];return A
def list_files_with_name(filename):
	A=[];D=[paths.script_path]+[A.path for A in extensions.active()]
	for B in D:
		if not os.path.isdir(B):continue
		C=os.path.join(B,filename)
		if os.path.isfile(C):A.append(C)
	return A
def load_scripts():
	B='.py';global current_basedir;scripts_data.clear();postprocessing_scripts_data.clear();script_callbacks.clear_callbacks();C=list_scripts('scripts',B)+list_scripts('modules/processing_scripts',B,include_extensions=_C);D=sys.path
	def E(module):
		C=module
		for B in C.__dict__.values():
			if not inspect.isclass(B):continue
			if issubclass(B,Script):scripts_data.append(ScriptClassData(B,A.path,A.basedir,C))
			elif issubclass(B,scripts_postprocessing.ScriptPostprocessing):postprocessing_scripts_data.append(ScriptClassData(B,A.path,A.basedir,C))
	def F(basedir):
		A={os.path.join(paths.script_path,'extensions-builtin'):1,paths.script_path:0}
		for B in A:
			if basedir.startswith(B):return A[B]
		return 9999
	for A in sorted(C,key=lambda x:[F(x.basedir),x]):
		try:
			if A.basedir!=paths.script_path:sys.path=[A.basedir]+sys.path
			current_basedir=A.basedir;G=script_loading.load_module(A.path);E(G)
		except Exception:errors.report(f"Error loading script: {A.filename}",exc_info=_B)
		finally:sys.path=D;current_basedir=paths.script_path;timer.startup_timer.record(A.filename)
	global scripts_txt2img,scripts_img2img,scripts_postproc;scripts_txt2img=ScriptRunner();scripts_img2img=ScriptRunner();scripts_postproc=scripts_postprocessing.ScriptPostprocessingRunner()
def wrap_call(func,filename,funcname,*A,default=_A,**B):
	try:return func(*A,**B)
	except Exception:errors.report(f"Error calling: {filename}/{funcname}",exc_info=_B)
	return default
class ScriptRunner:
	def __init__(A):A.scripts=[];A.selectable_scripts=[];A.alwayson_scripts=[];A.titles=[];A.title_map={};A.infotext_fields=[];A.paste_field_names=[];A.inputs=[_A];A.on_before_component_elem_id={};'dict of callbacks to be called before an element is created; key=elem_id, value=list of callbacks';A.on_after_component_elem_id={};'dict of callbacks to be called after an element is created; key=elem_id, value=list of callbacks'
	def initialize_scripts(B,is_img2img):
		C=is_img2img;from modules import scripts_auto_postprocessing as F;B.scripts.clear();B.alwayson_scripts.clear();B.selectable_scripts.clear();G=F.create_auto_preprocessing_script_data()
		for D in G+scripts_data:
			A=D.script_class();A.filename=D.path;A.is_txt2img=not C;A.is_img2img=C;A.tabname=_D if C else _E;E=A.show(A.is_img2img)
			if E==AlwaysVisible:B.scripts.append(A);B.alwayson_scripts.append(A);A.alwayson=_B
			elif E:B.scripts.append(A);B.selectable_scripts.append(A)
		B.apply_on_before_component_callbacks()
	def apply_on_before_component_callbacks(A):
		for C in A.scripts:
			E=C.on_before_component_elem_id or[];F=C.on_after_component_elem_id or[]
			for(B,D)in E:
				if B not in A.on_before_component_elem_id:A.on_before_component_elem_id[B]=[]
				A.on_before_component_elem_id[B].append((D,C))
			for(B,D)in F:
				if B not in A.on_after_component_elem_id:A.on_after_component_elem_id[B]=[]
				A.on_after_component_elem_id[B].append((D,C))
			E.clear();F.clear()
	def create_script_ui(B,script):
		A=script;import modules.api.models as E;A.args_from=len(B.inputs);A.args_to=len(B.inputs);C=wrap_call(A.ui,A.filename,'ui',A.is_img2img)
		if C is _A:return
		A.name=wrap_call(A.title,A.filename,_F,default=A.filename).lower();F=[]
		for D in C:
			D.custom_script_source=os.path.basename(A.filename);G=E.ScriptArg(label=D.label or'')
			for H in('value','minimum','maximum','step','choices'):
				I=getattr(D,H,_A)
				if I is not _A:setattr(G,H,I)
			F.append(G)
		A.api_info=E.ScriptInfo(name=A.name,is_img2img=A.is_img2img,is_alwayson=A.alwayson,args=F)
		if A.infotext_fields is not _A:B.infotext_fields+=A.infotext_fields
		if A.paste_field_names is not _A:B.paste_field_names+=A.paste_field_names
		B.inputs+=C;A.args_to=len(B.inputs)
	def setup_ui_for_section(B,section,scriptlist=_A):
		C=scriptlist
		if C is _A:C=B.alwayson_scripts
		for A in C:
			if A.alwayson and A.section!=section:continue
			if A.create_group:
				with gr.Group(visible=A.alwayson)as D:B.create_script_ui(A)
				A.group=D
			else:B.create_script_ui(A)
	def prepare_ui(A):A.inputs=[_A]
	def setup_ui(A):
		D='Script';C='None';E=[wrap_call(A.title,A.filename,_F)or A.filename for A in A.scripts];A.title_map={A.lower():B for(A,B)in zip(E,A.scripts)};A.titles=[wrap_call(A.title,A.filename,_F)or f"{A.filename} [error]"for A in A.selectable_scripts];A.setup_ui_for_section(_A);B=gr.Dropdown(label=D,elem_id='script_list',choices=[C]+A.titles,value=C,type='index');A.inputs[0]=B;A.setup_ui_for_section(_A,A.selectable_scripts)
		def F(script_index):B=script_index;C=A.selectable_scripts[B-1]if B>0 else _A;return[gr.update(visible=C==A)for A in A.selectable_scripts]
		def G(title):
			"called when an initial value is set from ui-config.json to show script's UI components";B=title
			if B==C:return
			D=A.titles.index(B);A.selectable_scripts[D].group.visible=_B
		B.init_field=G;B.change(fn=F,inputs=[B],outputs=[A.group for A in A.selectable_scripts]);A.script_load_ctr=0
		def H(params):
			B=params.get(D,_A)
			if B:C=A.titles.index(B);E=C==A.script_load_ctr;A.script_load_ctr=(A.script_load_ctr+1)%len(A.titles);return gr.update(visible=E)
			else:return gr.update(visible=_C)
		A.infotext_fields.append((B,lambda x:gr.update(value=x.get(D,C))));A.infotext_fields.extend([(A.group,H)for A in A.selectable_scripts]);A.apply_on_before_component_callbacks();return A.inputs
	def run(D,p,*B):
		C=B[0]
		if C==0:return
		A=D.selectable_scripts[C-1]
		if A is _A:return
		E=B[A.args_from:A.args_to];F=A.run(p,*E);shared.total_tqdm.clear();return F
	def before_process(B,p):
		for A in B.alwayson_scripts:
			try:C=p.script_args[A.args_from:A.args_to];A.before_process(p,*C)
			except Exception:errors.report(f"Error running before_process: {A.filename}",exc_info=_B)
	def process(B,p):
		for A in B.alwayson_scripts:
			try:C=p.script_args[A.args_from:A.args_to];A.process(p,*C)
			except Exception:errors.report(f"Error running process: {A.filename}",exc_info=_B)
	def before_process_batch(B,p,**C):
		for A in B.alwayson_scripts:
			try:D=p.script_args[A.args_from:A.args_to];A.before_process_batch(p,*D,**C)
			except Exception:errors.report(f"Error running before_process_batch: {A.filename}",exc_info=_B)
	def after_extra_networks_activate(B,p,**C):
		for A in B.alwayson_scripts:
			try:D=p.script_args[A.args_from:A.args_to];A.after_extra_networks_activate(p,*D,**C)
			except Exception:errors.report(f"Error running after_extra_networks_activate: {A.filename}",exc_info=_B)
	def process_batch(B,p,**C):
		for A in B.alwayson_scripts:
			try:D=p.script_args[A.args_from:A.args_to];A.process_batch(p,*D,**C)
			except Exception:errors.report(f"Error running process_batch: {A.filename}",exc_info=_B)
	def postprocess(B,p,processed):
		for A in B.alwayson_scripts:
			try:C=p.script_args[A.args_from:A.args_to];A.postprocess(p,processed,*C)
			except Exception:errors.report(f"Error running postprocess: {A.filename}",exc_info=_B)
	def postprocess_batch(B,p,images,**C):
		for A in B.alwayson_scripts:
			try:D=p.script_args[A.args_from:A.args_to];A.postprocess_batch(p,*D,images=images,**C)
			except Exception:errors.report(f"Error running postprocess_batch: {A.filename}",exc_info=_B)
	def postprocess_batch_list(B,p,pp,**C):
		for A in B.alwayson_scripts:
			try:D=p.script_args[A.args_from:A.args_to];A.postprocess_batch_list(p,pp,*D,**C)
			except Exception:errors.report(f"Error running postprocess_batch_list: {A.filename}",exc_info=_B)
	def postprocess_image(B,p,pp):
		for A in B.alwayson_scripts:
			try:C=p.script_args[A.args_from:A.args_to];A.postprocess_image(p,pp,*C)
			except Exception:errors.report(f"Error running postprocess_image: {A.filename}",exc_info=_B)
	def before_component(B,component,**C):
		D=component
		for(E,A)in B.on_before_component_elem_id.get(C.get('elem_id'),[]):
			try:E(OnComponent(component=D))
			except Exception:errors.report(f"Error running on_before_component: {A.filename}",exc_info=_B)
		for A in B.scripts:
			try:A.before_component(D,**C)
			except Exception:errors.report(f"Error running before_component: {A.filename}",exc_info=_B)
	def after_component(C,component,**D):
		B=component
		for(E,A)in C.on_after_component_elem_id.get(B.elem_id,[]):
			try:E(OnComponent(component=B))
			except Exception:errors.report(f"Error running on_after_component: {A.filename}",exc_info=_B)
		for A in C.scripts:
			try:A.after_component(B,**D)
			except Exception:errors.report(f"Error running after_component: {A.filename}",exc_info=_B)
	def script(A,title):return A.title_map.get(title.lower())
	def reload_sources(A,cache):
		G=cache
		for(B,C)in list(enumerate(A.scripts)):
			H=C.args_from;I=C.args_to;E=C.filename;D=G.get(E,_A)
			if D is _A:D=script_loading.load_module(C.filename);G[E]=D
			for F in D.__dict__.values():
				if type(F)==type and issubclass(F,Script):A.scripts[B]=F();A.scripts[B].filename=E;A.scripts[B].args_from=H;A.scripts[B].args_to=I
	def before_hr(B,p):
		for A in B.alwayson_scripts:
			try:C=p.script_args[A.args_from:A.args_to];A.before_hr(p,*C)
			except Exception:errors.report(f"Error running before_hr: {A.filename}",exc_info=_B)
	def setup_scrips(B,p,*,is_ui=_B):
		for A in B.alwayson_scripts:
			if not is_ui and A.setup_for_ui_only:continue
			try:C=p.script_args[A.args_from:A.args_to];A.setup(p,*C)
			except Exception:errors.report(f"Error running setup: {A.filename}",exc_info=_B)
scripts_txt2img=_A
scripts_img2img=_A
scripts_postproc=_A
scripts_current=_A
def reload_script_body_only():A={};scripts_txt2img.reload_sources(A);scripts_img2img.reload_sources(A)
reload_scripts=load_scripts