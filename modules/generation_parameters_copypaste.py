_L='Negative prompt'
_K='override_settings_component'
_J='init_img'
_I='Hires resize-2'
_H='Prompt'
_G='fields'
_F='Hires resize-1'
_E='Size-2'
_D='Size-1'
_C='\n'
_B=False
_A=None
import base64,io,json,os,re,gradio as gr
from modules.paths import data_path
from modules import shared,ui_tempdir,script_callbacks,processing
from PIL import Image
re_param_code='\\s*([\\w ]+):\\s*("(?:\\\\.|[^\\\\"])+"|[^,]*)(?:,|$)'
re_param=re.compile(re_param_code)
re_imagesize=re.compile('^(\\d+)x(\\d+)$')
re_hypernet_hash=re.compile('\\(([0-9a-f]+)\\)$')
type_of_gr_update=type(gr.update())
paste_fields={}
registered_param_bindings=[]
class ParamBinding:
	def __init__(A,paste_button,tabname,source_text_component=_A,source_image_component=_A,source_tabname=_A,override_settings_component=_A,paste_field_names=_A):A.paste_button=paste_button;A.tabname=tabname;A.source_text_component=source_text_component;A.source_image_component=source_image_component;A.source_tabname=source_tabname;A.override_settings_component=override_settings_component;A.paste_field_names=paste_field_names or[]
def reset():paste_fields.clear();registered_param_bindings.clear()
def quote(text):
	A=text
	if','not in str(A)and _C not in str(A)and':'not in str(A):return A
	return json.dumps(A,ensure_ascii=_B)
def unquote(text):
	A=text
	if len(A)==0 or A[0]!='"'or A[-1]!='"':return A
	try:return json.loads(A)
	except Exception:return A
def image_from_url_text(filedata):
	C='data:image/png;base64,';D='is_file';A=filedata
	if A is _A:return
	if type(A)==list and A and type(A[0])==dict and A[0].get(D,_B):A=A[0]
	if type(A)==dict and A.get(D,_B):B=A['name'];E=ui_tempdir.check_tmp_file(shared.demo,B);assert E,'trying to open image file outside of allowed directories';B=B.rsplit('?',1)[0];return Image.open(B)
	if type(A)==list:
		if len(A)==0:return
		A=A[0]
	if A.startswith(C):A=A[len(C):]
	A=base64.decodebytes(A.encode('utf-8'));F=Image.open(io.BytesIO(A));return F
def add_paste_fields(tabname,init_img,fields,override_settings_component=_A):
	A=fields;B=tabname;paste_fields[B]={_J:init_img,_G:A,_K:override_settings_component};import modules.ui
	if B=='txt2img':modules.ui.txt2img_paste_fields=A
	elif B=='img2img':modules.ui.img2img_paste_fields=A
def create_buttons(tabs_list):
	B={}
	for A in tabs_list:B[A]=gr.Button(f"Send to {A}",elem_id=f"{A}_tab")
	return B
def bind_buttons(buttons,send_image,send_generate_info):
	'old function for backwards compatibility; do not use this, use register_paste_params_button';A=send_generate_info
	for(B,C)in buttons.items():D=A if isinstance(A,gr.components.Component)else _A;E=A if isinstance(A,str)else _A;register_paste_params_button(ParamBinding(paste_button=C,tabname=B,source_text_component=D,source_image_component=send_image,source_tabname=E))
def register_paste_params_button(binding):registered_param_bindings.append(binding)
def connect_paste_params_buttons():
	A:0
	for A in registered_param_bindings:
		D=paste_fields[A.tabname][_J];B=paste_fields[A.tabname][_G];H=A.override_settings_component or paste_fields[A.tabname][_K];C=next(iter([A for(A,B)in B if B==_D]if B else[]),_A);I=next(iter([A for(A,B)in B if B==_E]if B else[]),_A)
		if A.source_image_component and D:
			if isinstance(A.source_image_component,gr.Gallery):E=send_image_and_dimensions if C else image_from_url_text;F='extract_image_from_gallery'
			else:E=send_image_and_dimensions if C else lambda x:x;F=_A
			A.paste_button.click(fn=E,_js=F,inputs=[A.source_image_component],outputs=[D,C,I]if C else[D],show_progress=_B)
		if A.source_text_component is not _A and B is not _A:connect_paste(A.paste_button,B,A.source_text_component,H,A.tabname)
		if A.source_tabname is not _A and B is not _A:G=[_H,_L,'Steps','Face restoration']+(['Seed']if shared.opts.send_seed else[])+A.paste_field_names;A.paste_button.click(fn=lambda*A:A,inputs=[A for(A,B)in paste_fields[A.source_tabname][_G]if B in G],outputs=[A for(A,B)in B if B in G],show_progress=_B)
		A.paste_button.click(fn=_A,_js=f"switch_to_{A.tabname}",inputs=_A,outputs=_A,show_progress=_B)
def send_image_and_dimensions(x):
	if isinstance(x,Image.Image):A=x
	else:A=image_from_url_text(x)
	if shared.opts.send_size and isinstance(A,Image.Image):B=A.width;C=A.height
	else:B=gr.update();C=gr.update()
	return A,B,C
def restore_old_hires_fix_params(res):
	'for infotexts that specify old First pass size parameter, convert it into\n    width, height, and hr scale';A=res;B=A.get('First pass size-1',_A);C=A.get('First pass size-2',_A)
	if shared.opts.use_old_hires_fix_width_height:
		D=int(A.get(_F,0));E=int(A.get(_I,0))
		if D and E:A[_D]=D;A[_E]=E;return
	if B is _A or C is _A:return
	B,C=int(B),int(C);F=int(A.get(_D,512));G=int(A.get(_E,512))
	if B==0 or C==0:B,C=processing.old_hires_fix_first_pass_dimensions(F,G)
	A[_D]=B;A[_E]=C;A[_F]=F;A[_I]=G
def parse_generation_parameters(x):
	"parses generation parameters string, the one you see in text field under the picture in UI:\n```\ngirl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate\nNegative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b\n```\n\n    returns a dict with field values\n    ";K='VAE Decoder';L='Full';M='VAE Encoder';N='Schedule rho';O='Schedule min sigma';P='Schedule max sigma';Q='Schedule type';R='RNG';S='Hires negative prompt';T='Hires prompt';U='Hires checkpoint';V='Hires sampler';W='Clip skip';X='Styles array';B='';A={};E=B;F=B;Y=_B;*Z,G=x.strip().split(_C)
	if len(re_param.findall(G))<3:Z.append(G);G=B
	for C in Z:
		C=C.strip()
		if C.startswith('Negative prompt:'):Y=True;C=C[16:].strip()
		if Y:F+=(B if F==B else _C)+C
		else:E+=(B if E==B else _C)+C
	if shared.opts.infotext_styles!='Ignore':
		I,E,F=shared.prompt_styles.extract_styles_from_prompt(E,F)
		if shared.opts.infotext_styles=='Apply':A[X]=I
		elif shared.opts.infotext_styles=='Apply if any'and I:A[X]=I
	A[_H]=E;A[_L]=F
	for(H,D)in re_param.findall(G):
		try:
			if D[0]=='"'and D[-1]=='"':D=unquote(D)
			J=re_imagesize.match(D)
			if J is not _A:A[f"{H}-1"]=J.group(1);A[f"{H}-2"]=J.group(2)
			else:A[H]=D
		except Exception:print(f'Error parsing "{H}: {D}"')
	if W not in A:A[W]='1'
	a=A.get('Hypernet',_A)
	if a is not _A:A[_H]+=f"<hypernet:{a}:{A.get('Hypernet strength','1.0')}>"
	if _F not in A:A[_F]=0;A[_I]=0
	if V not in A:A[V]='Use same sampler'
	if U not in A:A[U]='Use same checkpoint'
	if T not in A:A[T]=B
	if S not in A:A[S]=B
	restore_old_hires_fix_params(A)
	if R not in A:A[R]='GPU'
	if Q not in A:A[Q]='Automatic'
	if P not in A:A[P]=0
	if O not in A:A[O]=0
	if N not in A:A[N]=0
	if M not in A:A[M]=L
	if K not in A:A[K]=L
	return A
infotext_to_setting_name_mapping=[]
"Mapping of infotext labels to setting names. Only left for backwards compatibility - use OptionInfo(..., infotext='...') instead.\nExample content:\n\ninfotext_to_setting_name_mapping = [\n    ('Conditional mask weight', 'inpainting_mask_weight'),\n    ('Model hash', 'sd_model_checkpoint'),\n    ('ENSD', 'eta_noise_seed_delta'),\n    ('Schedule type', 'k_sched_type'),\n]\n"
def create_override_settings_dict(text_pairs):
	"creates processing's override_settings parameters from gradio's multiselect\n\n    Example input:\n        ['Clip skip: 2', 'Model hash: e6e99610c4', 'ENSD: 31337']\n\n    Example output:\n        {'CLIP_stop_at_last_layers': 2, 'sd_model_checkpoint': 'e6e99610c4', 'eta_noise_seed_delta': 31337}\n    ";A={};B={}
	for E in text_pairs:F,G=E.split(':',maxsplit=1);B[F]=G.strip()
	H=[(A.infotext,B)for(B,A)in shared.opts.data_labels.items()if A.infotext]
	for(I,C)in H+infotext_to_setting_name_mapping:
		D=B.get(I,_A)
		if D is _A:continue
		A[C]=shared.opts.cast_value(C,D)
	return A
def connect_paste(button,paste_fields,input_comp,override_settings_component,tabname):
	A=override_settings_component;C=button;B=paste_fields
	def D(prompt):
		D=prompt
		if not D and not shared.cmd_opts.hide_ui_dir_config:
			G=os.path.join(data_path,'params.txt')
			if os.path.exists(G):
				with open(G,'r',encoding='utf8')as J:D=J.read()
		E=parse_generation_parameters(D);script_callbacks.infotext_pasted_callback(D,E);C=[]
		for(K,F)in B:
			if callable(F):A=F(E)
			else:A=E.get(F,_A)
			if A is _A:C.append(gr.update())
			elif isinstance(A,type_of_gr_update):C.append(A)
			else:
				try:
					H=type(K.value)
					if H==bool and A=='False':I=_B
					else:I=H(A)
					C.append(gr.update(value=I))
				except Exception:C.append(gr.update())
		return C
	if A is not _A:
		F={A:1 for(B,A)in B}
		def E(params):
			E={};G=[(A.infotext,B)for(B,A)in shared.opts.data_labels.items()if A.infotext]
			for(B,C)in G+infotext_to_setting_name_mapping:
				if B in F:continue
				A=params.get(B,_A)
				if A is _A:continue
				if C=='sd_model_checkpoint'and shared.opts.disable_weights_auto_swap:continue
				A=shared.opts.cast_value(C,A);H=getattr(shared.opts,C,_A)
				if A==H:continue
				E[B]=A
			D=[f"{A}: {B}"for(A,B)in E.items()];return gr.Dropdown.update(value=D,choices=D,visible=bool(D))
		B=B+[(A,E)]
	C.click(fn=D,inputs=[input_comp],outputs=[A[0]for A in B],show_progress=_B);C.click(fn=_A,_js=f"recalculate_prompts_{tabname}",inputs=[],outputs=[],show_progress=_B)