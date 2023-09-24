_Q='visible'
_P='default'
_O='sd_model_checkpoint'
_N='Interrupt'
_M='RGB'
_L='image'
_K='Negative prompt'
_J='prompt'
_I='Prompt'
_H='ignore'
_G='primary'
_F='txt2img'
_E='compact'
_D='img2img'
_C=True
_B=None
_A=False
import datetime,mimetypes,os,sys
from functools import reduce
import warnings,gradio as gr,gradio.utils,numpy as np
from PIL import Image,PngImagePlugin
from modules.call_queue import wrap_gradio_gpu_call,wrap_queued_call,wrap_gradio_call
from modules import gradio_extensons
from modules import sd_hijack,sd_models,script_callbacks,ui_extensions,deepbooru,extra_networks,ui_common,ui_postprocessing,progress,ui_loadsave,shared_items,ui_settings,timer,sysinfo,ui_checkpoint_merger,ui_prompt_styles,scripts,sd_samplers,processing,ui_extra_networks
from modules.ui_components import FormRow,FormGroup,ToolButton,FormHTML,InputAccordion,ResizeHandleRow
from modules.paths import script_path
from modules.ui_common import create_refresh_button
from modules.ui_gradio_extensions import reload_javascript
from modules.shared import opts,cmd_opts
import modules.generation_parameters_copypaste as parameters_copypaste,modules.hypernetworks.ui as hypernetworks_ui,modules.textual_inversion.ui as textual_inversion_ui,modules.textual_inversion.textual_inversion as textual_inversion,modules.shared as shared,modules.images
from modules import prompt_parser
from modules.sd_hijack import model_hijack
from modules.generation_parameters_copypaste import image_from_url_text
create_setting_component=ui_settings.create_setting_component
warnings.filterwarnings(_P if opts.show_warnings else _H,category=UserWarning)
warnings.filterwarnings(_P if opts.show_gradio_deprecation_warnings else _H,category=gr.deprecation.GradioDeprecationWarning)
mimetypes.init()
mimetypes.add_type('application/javascript','.js')
mimetypes.add_type('image/webp','.webp')
if not cmd_opts.share and not cmd_opts.listen:gradio.utils.version_check=lambda:_B;gradio.utils.get_local_ip_address=lambda:'127.0.0.1'
if cmd_opts.ngrok is not _B:import modules.ngrok as ngrok;print('ngrok authtoken detected, trying to connect...');ngrok.connect(cmd_opts.ngrok,cmd_opts.port if cmd_opts.port is not _B else 7860,cmd_opts.ngrok_options)
def gr_show(visible=_C):return{_Q:visible,'__type__':'update'}
sample_img2img='assets/stable-samples/img2img/sketch-mountains-input.jpg'
sample_img2img=sample_img2img if os.path.exists(sample_img2img)else _B
random_symbol='🎲️'
reuse_symbol='♻️'
paste_symbol='↙️'
refresh_symbol='🔄'
save_style_symbol='💾'
apply_style_symbol='📋'
clear_prompt_symbol='🗑️'
extra_networks_symbol='🎴'
switch_values_symbol='⇅'
restore_progress_symbol='🌀'
detect_image_size_symbol='📐'
plaintext_to_html=ui_common.plaintext_to_html
def send_gradio_gallery_to_image(x):
	if len(x)==0:return
	return image_from_url_text(x[0])
def calc_resolution_hires(enable,width,height,hr_scale,hr_resize_x,hr_resize_y):
	if not enable:return''
	A=processing.StandardDemoProcessingTxt2Img(width=width,height=height,enable_hr=_C,hr_scale=hr_scale,hr_resize_x=hr_resize_x,hr_resize_y=hr_resize_y);A.calculate_target_resolution();return f"from <span class='resolution'>{A.width}x{A.height}</span> to <span class='resolution'>{A.hr_resize_x or A.hr_upscale_to_x}x{A.hr_resize_y or A.hr_upscale_to_y}</span>"
def resize_from_to_html(width,height,scale_by):
	A=scale_by;B=height;C=width;D=int(C*A);E=int(B*A)
	if not D or not E:return'no image selected'
	return f"resize: from <span class='resolution'>{C}x{B}</span> to <span class='resolution'>{D}x{E}</span>"
def process_interrogate(interrogation_function,mode,ii_input_dir,ii_output_dir,*D):
	E=ii_input_dir;C=interrogation_function;B=ii_output_dir;A=mode
	if A in{0,1,3,4}:return[C(D[A]),_B]
	elif A==2:return[C(D[A][_L]),_B]
	elif A==5:
		assert not shared.cmd_opts.hide_ui_dir_config,'Launched with --hide-ui-dir-config, batch img2img disabled';F=shared.listfiles(E);print(f"Will process {len(F)} images.")
		if B!='':os.makedirs(B,exist_ok=_C)
		else:B=E
		for G in F:H=Image.open(G);I=os.path.basename(G);J,K=os.path.splitext(I);print(C(H),file=open(os.path.join(B,f"{J}.txt"),'a',encoding='utf-8'))
		return[gr.update(),_B]
def interrogate(image):A=shared.interrogator.interrogate(image.convert(_M));return gr.update()if A is _B else A
def interrogate_deepbooru(image):A=deepbooru.model.tag(image);return gr.update()if A is _B else A
def connect_clear_prompt(button):'Given clear button, prompt, and token_counter objects, setup clear prompt button click event';button.click(_js='clear_prompt',fn=_B,inputs=[],outputs=[])
def update_token_counter(text,steps):
	B=steps;A=text
	try:A,C=extra_networks.parse_prompt(A);C,E,C=prompt_parser.get_multicond_prompt_list([A]);D=prompt_parser.get_learned_conditioning_prompt_schedules(E,B)
	except Exception:D=[[[B,A]]]
	F=reduce(lambda list1,list2:list1+list2,D);G=[A for(B,A)in F];H,I=max([model_hijack.get_prompt_lengths(A)for A in G],key=lambda args:args[0]);return f"<span class='gr-box gr-text-input'>{H}/{I}</span>"
class Toprow:
	'Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation'
	def __init__(A,is_img2img):
		C='token-counter';D='<span>0/75</span>';E=is_img2img;B=_D if E else _F;A.id_part=B
		with gr.Row(elem_id=f"{B}_toprow",variant=_E):
			with gr.Column(elem_id=f"{B}_prompt_container",scale=6):
				with gr.Row():
					with gr.Column(scale=80):
						with gr.Row():A.prompt=gr.Textbox(label=_I,elem_id=f"{B}_prompt",show_label=_A,lines=3,placeholder='Prompt (press Ctrl+Enter or Alt+Enter to generate)',elem_classes=[_J]);A.prompt_img=gr.File(label='',elem_id=f"{B}_prompt_image",file_count='single',type='binary',visible=_A)
				with gr.Row():
					with gr.Column(scale=80):
						with gr.Row():A.negative_prompt=gr.Textbox(label=_K,elem_id=f"{B}_neg_prompt",show_label=_A,lines=3,placeholder='Negative prompt (press Ctrl+Enter or Alt+Enter to generate)',elem_classes=[_J])
			A.button_interrogate=_B;A.button_deepbooru=_B
			if E:
				with gr.Column(scale=1,elem_classes='interrogate-col'):A.button_interrogate=gr.Button('Interrogate\nCLIP',elem_id='interrogate');A.button_deepbooru=gr.Button('Interrogate\nDeepBooru',elem_id='deepbooru')
			with gr.Column(scale=1,elem_id=f"{B}_actions_column"):
				with gr.Row(elem_id=f"{B}_generate_box",elem_classes='generate-box'):A.interrupt=gr.Button(_N,elem_id=f"{B}_interrupt",elem_classes='generate-box-interrupt');A.skip=gr.Button('Skip',elem_id=f"{B}_skip",elem_classes='generate-box-skip');A.submit=gr.Button('Generate',elem_id=f"{B}_generate",variant=_G);A.skip.click(fn=lambda:shared.state.skip(),inputs=[],outputs=[]);A.interrupt.click(fn=lambda:shared.state.interrupt(),inputs=[],outputs=[])
				with gr.Row(elem_id=f"{B}_tools"):A.paste=ToolButton(value=paste_symbol,elem_id='paste');A.clear_prompt_button=ToolButton(value=clear_prompt_symbol,elem_id=f"{B}_clear_prompt");A.restore_progress_button=ToolButton(value=restore_progress_symbol,elem_id=f"{B}_restore_progress",visible=_A);A.token_counter=gr.HTML(value=D,elem_id=f"{B}_token_counter",elem_classes=[C]);A.token_button=gr.Button(visible=_A,elem_id=f"{B}_token_button");A.negative_token_counter=gr.HTML(value=D,elem_id=f"{B}_negative_token_counter",elem_classes=[C]);A.negative_token_button=gr.Button(visible=_A,elem_id=f"{B}_negative_token_button");A.clear_prompt_button.click(fn=lambda*A:A,_js='confirm_clear_prompt',inputs=[A.prompt,A.negative_prompt],outputs=[A.prompt,A.negative_prompt])
				A.ui_styles=ui_prompt_styles.UiPromptStyles(B,A.prompt,A.negative_prompt)
		A.prompt_img.change(fn=modules.images.image_data,inputs=[A.prompt_img],outputs=[A.prompt,A.prompt_img],show_progress=_A)
def setup_progressbar(*A,**B):0
def apply_setting(key,value):
	B=value;A=key
	if B is _B:return gr.update()
	if shared.cmd_opts.freeze_settings:return gr.update()
	if A==_O and opts.disable_weights_auto_swap:return gr.update()
	if A==_O:
		D=sd_models.get_closet_checkpoint_match(B)
		if D is not _B:B=D.title
		else:return gr.update()
	C=opts.data_labels[A].component_args
	if C and isinstance(C,dict)and C.get(_Q)is _A:return
	E=type(opts.data_labels[A].default);F=opts.data.get(A,_B);opts.data[A]=E(B)if E!=type(_B)else B
	if F!=B and opts.data_labels[A].onchange is not _B:opts.data_labels[A].onchange()
	opts.save(shared.config_filename);return getattr(opts,A)
def create_output_panel(tabname,outdir):return ui_common.create_output_panel(tabname,outdir)
def create_sampler_and_steps_selection(choices,tabname):
	C='Sampling steps';D='Sampling method';B=choices;A=tabname
	if opts.samplers_in_dropdown:
		with FormRow(elem_id=f"sampler_selection_{A}"):E=gr.Dropdown(label=D,elem_id=f"{A}_sampling",choices=B,value=B[0]);F=gr.Slider(minimum=1,maximum=150,step=1,elem_id=f"{A}_steps",label=C,value=20)
	else:
		with FormGroup(elem_id=f"sampler_selection_{A}"):F=gr.Slider(minimum=1,maximum=150,step=1,elem_id=f"{A}_steps",label=C,value=20);E=gr.Radio(label=D,elem_id=f"{A}_sampling",choices=B,value=B[0])
	return F,E
def ordered_ui_categories():
	A={B.strip():A*2+1 for(A,B)in enumerate(shared.opts.ui_reorder_list)}
	for(C,B)in sorted(enumerate(shared_items.ui_reorder_categories()),key=lambda x:A.get(x[1],x[0]*2+0)):yield B
def create_override_settings_dropdown(tabname,row):A=gr.Dropdown([],label='Override settings',visible=_A,elem_id=f"{tabname}_override_settings",multiselect=_C);A.change(fn=lambda x:gr.Dropdown.update(visible=bool(x)),inputs=[A],outputs=[A]);return A
def create_ui():
	AX='notification.mp3';AY='extensions';AZ='settings';Aa='disabled';Ab='Hypernetwork Learning rate';Ac='Embedding Learning rate';Ad='Maximize area';Ae='Normal';Af='Create hypernetwork';Ag='Create embedding';Ah='extras';Ai='Whole picture';Aj='original';Ak='Inpaint masked';Al='Mask blur';Am='img2img_batch_size';An='img2img_batch_count';Ao='img2img_column_batch';Ap='currentImg2imgSourceResolution';Aq='img2img_column_size';Ar='Just resize';As='color-sketch';At='img2img_sketch';Au='Hires sampler';Av='Hires resize-1';Aw='Hires upscaler';Ax='Hires upscale';Ay='Size-2';Az='Size-1';A_='scripts';B0='override_settings';B1='Hires steps';B2='checkboxes-row';B3='checkboxes';B4='CFG Scale';B5='txt2img_batch_size';B6='txt2img_batch_count';B7='txt2img_column_batch';B8='dimensions-tools';B9='dimensions';BA='sampler';BB='Generation';z='start_training_textual_inversion';A0='RGBA';A1='Image for img2img';A2='CFG scale';A3='Sampler';A4='Steps';A5='batch';A6='Hires negative prompt';A7='Hires prompt';A8='Hires checkpoint';h='index';i='inpaint_sketch';j='sketch';k='Styles array';l='Use same sampler';m='choices';n='Batch count';o='Height';p='Width';U='Use same checkpoint';V='Denoising strength';O='accordions';K='pil';L='upload';M='inpaint';N='Batch size';G=.0;E=1.;import modules.img2img,modules.txt2img;reload_javascript();parameters_copypaste.reset();scripts.scripts_current=scripts.scripts_txt2img;scripts.scripts_txt2img.initialize_scripts(is_img2img=_A)
	with gr.Blocks(analytics_enabled=_A)as BC:
		A=Toprow(is_img2img=_A);C=gr.Label(visible=_A);W=gr.Tabs(elem_id='txt2img_extra_tabs');W.__enter__()
		with gr.Tab(BB,id='txt2img_generation')as Bs,ResizeHandleRow(equal_height=_A):
			with gr.Column(variant=_E,elem_id='txt2img_settings'):
				scripts.scripts_txt2img.prepare_ui()
				for B in ordered_ui_categories():
					if B==BA:D,P=create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(),_F)
					elif B==B9:
						with FormRow():
							with gr.Column(elem_id='txt2img_column_size',scale=4):H=gr.Slider(minimum=64,maximum=2048,step=8,label=p,value=512,elem_id='txt2img_width');I=gr.Slider(minimum=64,maximum=2048,step=8,label=o,value=512,elem_id='txt2img_height')
							with gr.Column(elem_id='txt2img_dimensions_row',scale=1,elem_classes=B8):A9=ToolButton(value=switch_values_symbol,elem_id='txt2img_res_switch_btn',label='Switch dims')
							if opts.dimensions_and_batch_together:
								with gr.Column(elem_id=B7):X=gr.Slider(minimum=1,step=1,label=n,value=1,elem_id=B6);F=gr.Slider(minimum=1,maximum=8,step=1,label=N,value=1,elem_id=B5)
					elif B=='cfg':
						with gr.Row():Q=gr.Slider(minimum=E,maximum=3e1,step=.5,label=B4,value=7.,elem_id='txt2img_cfg_scale')
					elif B==B3:
						with FormRow(elem_classes=B2,variant=_E):0
					elif B==O:
						with gr.Row(elem_id='txt2img_accordions',elem_classes=O):
							with InputAccordion(_A,label='Hires. fix',elem_id='txt2img_hr')as q:
								with q.extra():Bt=FormHTML(value='',elem_id='txtimg_hr_finalres',label='Upscaled resolution',interactive=_A,min_width=0)
								with FormRow(elem_id='txt2img_hires_fix_row1',variant=_E):BD=gr.Dropdown(label='Upscaler',elem_id='txt2img_hr_upscaler',choices=[*shared.latent_upscale_modes,*[A.name for A in shared.sd_upscalers]],value=shared.latent_upscale_default_mode);BE=gr.Slider(minimum=0,maximum=150,step=1,label=B1,value=0,elem_id='txt2img_hires_steps');Y=gr.Slider(minimum=G,maximum=E,step=.01,label=V,value=.7,elem_id='txt2img_denoising_strength')
								with FormRow(elem_id='txt2img_hires_fix_row2',variant=_E):AA=gr.Slider(minimum=E,maximum=4.,step=.05,label='Upscale by',value=2.,elem_id='txt2img_hr_scale');AB=gr.Slider(minimum=0,maximum=2048,step=8,label='Resize width to',value=0,elem_id='txt2img_hr_resize_x');AC=gr.Slider(minimum=0,maximum=2048,step=8,label='Resize height to',value=0,elem_id='txt2img_hr_resize_y')
								with FormRow(elem_id='txt2img_hires_fix_row3',variant=_E,visible=opts.hires_fix_show_sampler)as Bu:AD=gr.Dropdown(label=A8,elem_id='hr_checkpoint',choices=[U]+modules.sd_models.checkpoint_tiles(use_short=_C),value=U);create_refresh_button(AD,modules.sd_models.list_models,lambda:{m:[U]+modules.sd_models.checkpoint_tiles(use_short=_C)},'hr_checkpoint_refresh');BF=gr.Dropdown(label='Hires sampling method',elem_id='hr_sampler',choices=[l]+sd_samplers.visible_sampler_names(),value=l)
								with FormRow(elem_id='txt2img_hires_fix_row4',variant=_E,visible=opts.hires_fix_show_prompts)as Bv:
									with gr.Column(scale=80):
										with gr.Row():BG=gr.Textbox(label=A7,elem_id='hires_prompt',show_label=_A,lines=3,placeholder='Prompt for hires fix pass.\nLeave empty to use the same prompt as in first pass.',elem_classes=[_J])
									with gr.Column(scale=80):
										with gr.Row():BH=gr.Textbox(label=A6,elem_id='hires_neg_prompt',show_label=_A,lines=3,placeholder='Negative prompt for hires fix pass.\nLeave empty to use the same negative prompt as in first pass.',elem_classes=[_J])
							scripts.scripts_txt2img.setup_ui_for_section(B)
					elif B==A5:
						if not opts.dimensions_and_batch_together:
							with FormRow(elem_id=B7):X=gr.Slider(minimum=1,step=1,label=n,value=1,elem_id=B6);F=gr.Slider(minimum=1,maximum=8,step=1,label=N,value=1,elem_id=B5)
					elif B==B0:
						with FormRow(elem_id='txt2img_override_settings_row')as AE:R=create_override_settings_dropdown(_F,AE)
					elif B==A_:
						with FormGroup(elem_id='txt2img_script_container'):AF=scripts.scripts_txt2img.setup_ui()
					if B not in{O}:scripts.scripts_txt2img.setup_ui_for_section(B)
			AG=[q,H,I,AA,AB,AC]
			for Z in AG:BI=Z.release if isinstance(Z,gr.Slider)else Z.change;BI(fn=calc_resolution_hires,inputs=AG,outputs=[Bt],show_progress=_A);BI(_B,_js='onCalcResolutionHires',inputs=AG,outputs=[],show_progress=_A)
			AH,J,a,b=create_output_panel(_F,opts.outdir_txt2img_samples);BJ=dict(fn=wrap_gradio_gpu_call(modules.txt2img.txt2img,extra_outputs=[_B,'','']),_js='submit',inputs=[C,A.prompt,A.negative_prompt,A.ui_styles.dropdown,D,P,X,F,Q,I,H,q,Y,AA,BD,BE,AB,AC,AD,BF,BG,BH,R]+AF,outputs=[AH,J,a,b],show_progress=_A);A.prompt.submit(**BJ);A.submit.click(**BJ);A9.click(fn=_B,_js="function(){switchWidthHeight('txt2img')}",inputs=_B,outputs=_B,show_progress=_A);A.restore_progress_button.click(fn=progress.restore_progress,_js='restoreProgressTxt2img',inputs=[C],outputs=[AH,J,a,b],show_progress=_A);Bw=[(A.prompt,_I),(A.negative_prompt,_K),(D,A4),(P,A3),(Q,A2),(H,Az),(I,Ay),(F,N),(A.ui_styles.dropdown,lambda d:d[k]if isinstance(d.get(k),list)else gr.update()),(Y,V),(q,lambda d:V in d and(Ax in d or Aw in d or Av in d)),(AA,Ax),(BD,Aw),(BE,B1),(AB,Av),(AC,'Hires resize-2'),(AD,A8),(BF,Au),(Bu,lambda d:gr.update(visible=_C)if d.get(Au,l)!=l or d.get(A8,U)!=U else gr.update()),(BG,A7),(BH,A6),(Bv,lambda d:gr.update(visible=_C)if d.get(A7,'')!=''or d.get(A6,'')!=''else gr.update()),*scripts.scripts_txt2img.infotext_fields];parameters_copypaste.add_paste_fields(_F,_B,Bw,R);parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=A.paste,tabname=_F,source_text_component=A.prompt,source_image_component=_B));AI=[A.prompt,A.negative_prompt,D,P,Q,scripts.scripts_txt2img.script('Seed').seed,H,I];A.token_button.click(fn=wrap_queued_call(update_token_counter),inputs=[A.prompt,D],outputs=[A.token_counter]);A.negative_token_button.click(fn=wrap_queued_call(update_token_counter),inputs=[A.negative_prompt,D],outputs=[A.negative_token_counter])
		Bx=ui_extra_networks.create_ui(BC,[Bs],_F);ui_extra_networks.setup_ui(Bx,AH);W.__exit__()
	scripts.scripts_current=scripts.scripts_img2img;scripts.scripts_img2img.initialize_scripts(is_img2img=_C)
	with gr.Blocks(analytics_enabled=_A)as BK:
		A=Toprow(is_img2img=_C);W=gr.Tabs(elem_id='img2img_extra_tabs');W.__enter__()
		with gr.Tab(BB,id='img2img_generation')as By,ResizeHandleRow(equal_height=_A):
			with gr.Column(variant=_E,elem_id='img2img_settings'):
				BL=[];BM={}
				def r(tab_name,elem):
					A=tab_name
					with gr.Row(variant=_E,elem_id=f"img2img_copy_to_{A}"):
						gr.HTML('Copy image to: ',elem_id=f"img2img_label_copy_to_{A}")
						for(C,B)in zip([_D,j,M,'inpaint sketch'],[_D,j,M,i]):
							if B==A:gr.Button(C,interactive=_A);BM[B]=elem;continue
							D=gr.Button(C);BL.append((D,B,elem))
				with gr.Tabs(elem_id='mode_img2img'):
					Bz=gr.State(0)
					with gr.TabItem(_D,id=_D,elem_id='img2img_img2img_tab')as B_:c=gr.Image(label=A1,elem_id='img2img_image',show_label=_A,source=L,interactive=_C,type=K,tool='editor',image_mode=A0,height=opts.img2img_editor_height);r(_D,c)
					with gr.TabItem('Sketch',id=At,elem_id='img2img_img2img_sketch_tab')as C0:s=gr.Image(label=A1,elem_id=At,show_label=_A,source=L,interactive=_C,type=K,tool=As,image_mode=_M,height=opts.img2img_editor_height,brush_color=opts.img2img_sketch_default_brush_color);r(j,s)
					with gr.TabItem('Inpaint',id=M,elem_id='img2img_inpaint_tab')as C1:t=gr.Image(label='Image for inpainting with mask',show_label=_A,elem_id='img2maskimg',source=L,interactive=_C,type=K,tool=j,image_mode=A0,height=opts.img2img_editor_height,brush_color=opts.img2img_inpaint_mask_brush_color);r(M,t)
					with gr.TabItem('Inpaint sketch',id=i,elem_id='img2img_inpaint_sketch_tab')as C2:
						d=gr.Image(label='Color sketch inpainting',show_label=_A,elem_id=i,source=L,interactive=_C,type=K,tool=As,image_mode=_M,height=opts.img2img_editor_height,brush_color=opts.img2img_inpaint_sketch_default_brush_color);AJ=gr.State(_B);r(i,d)
						def C3(image,state):
							B=image;A=state
							if B is not _B:C=A is not _B and A.size==B.size;D=np.any(np.all(np.array(B)==np.array(A),axis=-1));E=C and D;return B if not E or A is _B else A
						d.change(C3,[d,AJ],AJ)
					with gr.TabItem('Inpaint upload',id='inpaint_upload',elem_id='img2img_inpaint_upload_tab')as C4:BN=gr.Image(label=A1,show_label=_A,source=L,interactive=_C,type=K,elem_id='img_inpaint_base');C5=gr.Image(label='Mask',source=L,interactive=_C,type=K,image_mode=A0,elem_id='img_inpaint_mask')
					with gr.TabItem('Batch',id=A5,elem_id='img2img_batch_tab')as C6:
						C7='<br>Disabled when launched with --hide-ui-dir-config.'if shared.cmd_opts.hide_ui_dir_config else'';gr.HTML('<p style=\'padding-bottom: 1em;\' class="text-gray-500">Process images in a directory on the same machine where the server is running.'+'<br>Use an empty output directory to save pictures normally instead of writing to the output directory.'+f"<br>Add inpaint batch mask directory to enable inpaint batch processing.{C7}</p>");BO=gr.Textbox(label='Input directory',**shared.hide_dirs,elem_id='img2img_batch_input_dir');BP=gr.Textbox(label='Output directory',**shared.hide_dirs,elem_id='img2img_batch_output_dir');C8=gr.Textbox(label='Inpaint batch mask directory (required for inpaint batch processing only)',**shared.hide_dirs,elem_id='img2img_batch_inpaint_mask_dir')
						with gr.Accordion('PNG info',open=_A):C9=gr.Checkbox(label='Append png info to prompts',**shared.hide_dirs,elem_id='img2img_batch_use_png_info');CA=gr.Textbox(label='PNG info directory',**shared.hide_dirs,placeholder='Leave empty to use input directory',elem_id='img2img_batch_png_info_dir');CB=gr.CheckboxGroup([_I,_K,'Seed',A2,A3,A4],label='Parameters to take from png info',info='Prompts from png info will be appended to prompts set in ui.')
					BQ=[B_,C0,C1,C2,C4,C6]
					for(AK,CC)in enumerate(BQ):CC.select(fn=lambda tabnum=AK:tabnum,inputs=[],outputs=[Bz])
				def CD(img):
					A=img
					if isinstance(A,dict)and _L in A:return A[_L]
					return A
				for(u,BR,AL)in BL:u.click(fn=CD,inputs=[AL],outputs=[BM[BR]]);u.click(fn=lambda:_B,_js=f"switch_to_{BR.replace(' ','_')}",inputs=[],outputs=[])
				with FormRow():CE=gr.Radio(label='Resize mode',elem_id='resize_mode',choices=[Ar,'Crop and resize','Resize and fill','Just resize (latent upscale)'],type=h,value=Ar)
				scripts.scripts_img2img.prepare_ui()
				for B in ordered_ui_categories():
					if B==BA:D,P=create_sampler_and_steps_selection(sd_samplers.visible_sampler_names(),_D)
					elif B==B9:
						with FormRow():
							with gr.Column(elem_id=Aq,scale=4):
								AM=gr.State(value=0)
								with gr.Tabs():
									with gr.Tab(label='Resize to',elem_id='img2img_tab_resize_to')as CF:
										with FormRow():
											with gr.Column(elem_id=Aq,scale=4):H=gr.Slider(minimum=64,maximum=2048,step=8,label=p,value=512,elem_id='img2img_width');I=gr.Slider(minimum=64,maximum=2048,step=8,label=o,value=512,elem_id='img2img_height')
											with gr.Column(elem_id='img2img_dimensions_row',scale=1,elem_classes=B8):A9=ToolButton(value=switch_values_symbol,elem_id='img2img_res_switch_btn');CG=ToolButton(value=detect_image_size_symbol,elem_id='img2img_detect_image_size_btn')
									with gr.Tab(label='Resize by',elem_id='img2img_tab_resize_by')as CH:
										AN=gr.Slider(minimum=.05,maximum=4.,step=.05,label='Scale',value=E,elem_id='img2img_scale')
										with FormRow():CI=FormHTML(resize_from_to_html(0,0,G),elem_id='img2img_scale_resolution_preview');gr.Slider(label='Unused',elem_id='img2img_unused_scale_by_slider');CJ=gr.Button(visible=_A,elem_id='img2img_update_resize_to')
									BS=dict(fn=resize_from_to_html,_js=Ap,inputs=[C,C,AN],outputs=CI,show_progress=_A);AN.release(**BS);CJ.click(**BS)
									for Z in[c,s]:Z.change(fn=lambda:_B,_js='updateImg2imgResizeToTextAfterChangingImage',inputs=[],outputs=[],show_progress=_A)
							CF.select(fn=lambda:0,inputs=[],outputs=[AM]);CH.select(fn=lambda:1,inputs=[],outputs=[AM])
							if opts.dimensions_and_batch_together:
								with gr.Column(elem_id=Ao):X=gr.Slider(minimum=1,step=1,label=n,value=1,elem_id=An);F=gr.Slider(minimum=1,maximum=8,step=1,label=N,value=1,elem_id=Am)
					elif B=='denoising':Y=gr.Slider(minimum=G,maximum=E,step=.01,label=V,value=.75,elem_id='img2img_denoising_strength')
					elif B=='cfg':
						with gr.Row():Q=gr.Slider(minimum=E,maximum=3e1,step=.5,label=B4,value=7.,elem_id='img2img_cfg_scale');v=gr.Slider(minimum=0,maximum=3.,step=.05,label='Image CFG Scale',value=1.5,elem_id='img2img_image_cfg_scale',visible=_A)
					elif B==B3:
						with FormRow(elem_classes=B2,variant=_E):0
					elif B==O:
						with gr.Row(elem_id='img2img_accordions',elem_classes=O):scripts.scripts_img2img.setup_ui_for_section(B)
					elif B==A5:
						if not opts.dimensions_and_batch_together:
							with FormRow(elem_id=Ao):X=gr.Slider(minimum=1,step=1,label=n,value=1,elem_id=An);F=gr.Slider(minimum=1,maximum=8,step=1,label=N,value=1,elem_id=Am)
					elif B==B0:
						with FormRow(elem_id='img2img_override_settings_row')as AE:R=create_override_settings_dropdown(_D,AE)
					elif B==A_:
						with FormGroup(elem_id='img2img_script_container'):AF=scripts.scripts_img2img.setup_ui()
					elif B==M:
						with FormGroup(elem_id='inpaint_controls',visible=_A)as CK:
							with FormRow():BT=gr.Slider(label=Al,minimum=0,maximum=64,step=1,value=4,elem_id='img2img_mask_blur');BU=gr.Slider(label='Mask transparency',visible=_A,elem_id='img2img_mask_alpha')
							with FormRow():CL=gr.Radio(label='Mask mode',choices=[Ak,'Inpaint not masked'],value=Ak,type=h,elem_id='img2img_mask_mode')
							with FormRow():CM=gr.Radio(label='Masked content',choices=['fill',Aj,'latent noise','latent nothing'],value=Aj,type=h,elem_id='img2img_inpainting_fill')
							with FormRow():
								with gr.Column():CN=gr.Radio(label='Inpaint area',choices=[Ai,'Only masked'],type=h,value=Ai,elem_id='img2img_inpaint_full_res')
								with gr.Column(scale=4):CO=gr.Slider(label='Only masked padding, pixels',minimum=0,maximum=256,step=4,value=32,elem_id='img2img_inpaint_full_res_padding')
							def CP(tab):return gr.update(visible=tab in[2,3,4]),gr.update(visible=tab==3)
							for(AK,AL)in enumerate(BQ):AL.select(fn=lambda tab=AK:CP(tab),inputs=[],outputs=[CK,BU])
					if B not in{O}:scripts.scripts_img2img.setup_ui_for_section(B)
			AO,J,a,b=create_output_panel(_D,opts.outdir_img2img_samples);BV=dict(fn=wrap_gradio_gpu_call(modules.img2img.img2img,extra_outputs=[_B,'','']),_js='submit_img2img',inputs=[C,C,A.prompt,A.negative_prompt,A.ui_styles.dropdown,c,s,t,d,AJ,BN,C5,D,P,BT,BU,CM,X,F,Q,v,Y,AM,I,H,AN,CE,CN,CO,CL,BO,BP,C8,R,C9,CB,CA]+AF,outputs=[AO,J,a,b],show_progress=_A);BW=dict(_js='get_img2img_tab_index',inputs=[C,BO,BP,c,s,t,d,BN],outputs=[A.prompt,C]);A.prompt.submit(**BV);A.submit.click(**BV);A9.click(fn=_B,_js="function(){switchWidthHeight('img2img')}",inputs=_B,outputs=_B,show_progress=_A);CG.click(fn=lambda w,h,_:(w or gr.update(),h or gr.update()),_js=Ap,inputs=[C,C,C],outputs=[H,I],show_progress=_A);A.restore_progress_button.click(fn=progress.restore_progress,_js='restoreProgressImg2img',inputs=[C],outputs=[AO,J,a,b],show_progress=_A);A.button_interrogate.click(fn=lambda*A:process_interrogate(interrogate,*A),**BW);A.button_deepbooru.click(fn=lambda*A:process_interrogate(interrogate_deepbooru,*A),**BW);A.token_button.click(fn=update_token_counter,inputs=[A.prompt,D],outputs=[A.token_counter]);A.negative_token_button.click(fn=wrap_queued_call(update_token_counter),inputs=[A.negative_prompt,D],outputs=[A.negative_token_counter]);BX=[(A.prompt,_I),(A.negative_prompt,_K),(D,A4),(P,A3),(Q,A2),(v,'Image CFG scale'),(H,Az),(I,Ay),(F,N),(A.ui_styles.dropdown,lambda d:d[k]if isinstance(d.get(k),list)else gr.update()),(Y,V),(BT,Al),*scripts.scripts_img2img.infotext_fields];parameters_copypaste.add_paste_fields(_D,c,BX,R);parameters_copypaste.add_paste_fields(M,t,BX,R);parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=A.paste,tabname=_D,source_text_component=A.prompt,source_image_component=_B))
		CQ=ui_extra_networks.create_ui(BK,[By],_D);ui_extra_networks.setup_ui(CQ,AO);W.__exit__()
	scripts.scripts_current=_B
	with gr.Blocks(analytics_enabled=_A)as CR:ui_postprocessing.create_ui()
	with gr.Blocks(analytics_enabled=_A)as CS:
		with gr.Row(equal_height=_A):
			with gr.Column(variant='panel'):AP=gr.Image(elem_id='pnginfo_image',label='Source',source=L,interactive=_C,type=K)
			with gr.Column(variant='panel'):
				CT=gr.HTML();J=gr.Textbox(visible=_A,elem_id='pnginfo_generation_info');CU=gr.HTML()
				with gr.Row():CV=parameters_copypaste.create_buttons([_F,_D,M,Ah])
				for(CW,u)in CV.items():parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=u,tabname=CW,source_text_component=J,source_image_component=AP))
		AP.change(fn=wrap_gradio_call(modules.extras.run_pnginfo),inputs=[AP],outputs=[CT,J,CU])
	BY=ui_checkpoint_merger.UiCheckpointMerger()
	with gr.Blocks(analytics_enabled=_A)as CX:
		with gr.Row(equal_height=_A):gr.HTML(value='<p style=\'margin-bottom: 0.7em\'>See <b><a href="https://github.com/tenebo/standard-demo-we/wiki/Textual-Inversion">wiki</a></b> for detailed explanation.</p>')
		with gr.Row(variant=_E,equal_height=_A):
			with gr.Tabs(elem_id='train_tabs'):
				with gr.Tab(label=Ag,id='create_embedding'):
					CY=gr.Textbox(label='Name',elem_id='train_new_embedding_name');CZ=gr.Textbox(label='Initialization text',value='*',elem_id='train_initialization_text');Ca=gr.Slider(label='Number of vectors per token',minimum=1,maximum=75,step=1,value=1,elem_id='train_nvpt');Cb=gr.Checkbox(value=_A,label='Overwrite Old Embedding',elem_id='train_overwrite_old_embedding')
					with gr.Row():
						with gr.Column(scale=3):gr.HTML(value='')
						with gr.Column():Cc=gr.Button(value=Ag,variant=_G,elem_id='train_create_embedding')
				with gr.Tab(label=Af,id='create_hypernetwork'):
					Cd=gr.Textbox(label='Name',elem_id='train_new_hypernetwork_name');Ce=gr.CheckboxGroup(label='Modules',value=['768','320','640','1280'],choices=['768','1024','320','640','1280'],elem_id='train_new_hypernetwork_sizes');Cf=gr.Textbox('1, 2, 1',label='Enter hypernetwork layer structure',placeholder="1st and last digit must be 1. ex:'1, 2, 1'",elem_id='train_new_hypernetwork_layer_structure');Cg=gr.Dropdown(value='linear',label='Select activation function of hypernetwork. Recommended : Swish / Linear(none)',choices=hypernetworks_ui.keys,elem_id='train_new_hypernetwork_activation_func');Ch=gr.Dropdown(value=Ae,label='Select Layer weights initialization. Recommended: Kaiming for relu-like, Xavier for sigmoid-like, Normal otherwise',choices=[Ae,'KaimingUniform','KaimingNormal','XavierUniform','XavierNormal'],elem_id='train_new_hypernetwork_initialization_option');Ci=gr.Checkbox(label='Add layer normalization',elem_id='train_new_hypernetwork_add_layer_norm');Cj=gr.Checkbox(label='Use dropout',elem_id='train_new_hypernetwork_use_dropout');Ck=gr.Textbox('0, 0, 0',label='Enter hypernetwork Dropout structure (or empty). Recommended : 0~0.35 incrementing sequence: 0, 0.05, 0.15',placeholder="1st and last digit must be 0 and values should be between 0 and 1. ex:'0, 0.01, 0'");Cl=gr.Checkbox(value=_A,label='Overwrite Old Hypernetwork',elem_id='train_overwrite_old_hypernetwork')
					with gr.Row():
						with gr.Column(scale=3):gr.HTML(value='')
						with gr.Column():Cm=gr.Button(value=Af,variant=_G,elem_id='train_create_hypernetwork')
				with gr.Tab(label='Preprocess images',id='preprocess_images'):
					Cn=gr.Textbox(label='Source directory',elem_id='train_process_src');Co=gr.Textbox(label='Destination directory',elem_id='train_process_dst');Cp=gr.Slider(minimum=64,maximum=2048,step=8,label=p,value=512,elem_id='train_process_width');Cq=gr.Slider(minimum=64,maximum=2048,step=8,label=o,value=512,elem_id='train_process_height');Cr=gr.Dropdown(label='Existing Caption txt Action',value=_H,choices=[_H,'copy','prepend','append'],elem_id='train_preprocess_txt_action')
					with gr.Row():Cs=gr.Checkbox(label='Keep original size',elem_id='train_process_keep_original_size');Ct=gr.Checkbox(label='Create flipped copies',elem_id='train_process_flip');AQ=gr.Checkbox(label='Split oversized images',elem_id='train_process_split');AR=gr.Checkbox(label='Auto focal point crop',elem_id='train_process_focal_crop');AS=gr.Checkbox(label='Auto-sized crop',elem_id='train_process_multicrop');Cu=gr.Checkbox(label='Use BLIP for caption',elem_id='train_process_caption');Cv=gr.Checkbox(label='Use deepbooru for caption',visible=_C,elem_id='train_process_caption_deepbooru')
					with gr.Row(visible=_A)as Cw:Cx=gr.Slider(label='Split image threshold',value=.5,minimum=G,maximum=E,step=.05,elem_id='train_process_split_threshold');Cy=gr.Slider(label='Split image overlap ratio',value=.2,minimum=G,maximum=.9,step=.05,elem_id='train_process_overlap_ratio')
					with gr.Row(visible=_A)as Cz:C_=gr.Slider(label='Focal point face weight',value=.9,minimum=G,maximum=E,step=.05,elem_id='train_process_focal_crop_face_weight');D0=gr.Slider(label='Focal point entropy weight',value=.15,minimum=G,maximum=E,step=.05,elem_id='train_process_focal_crop_entropy_weight');D1=gr.Slider(label='Focal point edges weight',value=.5,minimum=G,maximum=E,step=.05,elem_id='train_process_focal_crop_edges_weight');D2=gr.Checkbox(label='Create debug image',elem_id='train_process_focal_crop_debug')
					with gr.Column(visible=_A)as D3:
						gr.Markdown('Each image is center-cropped with an automatically chosen width and height.')
						with gr.Row():D4=gr.Slider(minimum=64,maximum=2048,step=8,label='Dimension lower bound',value=384,elem_id='train_process_multicrop_mindim');D5=gr.Slider(minimum=64,maximum=2048,step=8,label='Dimension upper bound',value=768,elem_id='train_process_multicrop_maxdim')
						with gr.Row():D6=gr.Slider(minimum=64*64,maximum=2048*2048,step=1,label='Area lower bound',value=64*64,elem_id='train_process_multicrop_minarea');D7=gr.Slider(minimum=64*64,maximum=2048*2048,step=1,label='Area upper bound',value=640*640,elem_id='train_process_multicrop_maxarea')
						with gr.Row():D8=gr.Radio([Ad,'Minimize error'],value=Ad,label='Resizing objective',elem_id='train_process_multicrop_objective');D9=gr.Slider(minimum=0,maximum=1,step=.01,label='Error threshold',value=.1,elem_id='train_process_multicrop_threshold')
					with gr.Row():
						with gr.Column(scale=3):gr.HTML(value='')
						with gr.Column():
							with gr.Row():DA=gr.Button(_N,elem_id='train_interrupt_preprocessing')
							DB=gr.Button(value='Preprocess',variant=_G,elem_id='train_run_preprocess')
					AQ.change(fn=lambda show:gr_show(show),inputs=[AQ],outputs=[Cw]);AR.change(fn=lambda show:gr_show(show),inputs=[AR],outputs=[Cz]);AS.change(fn=lambda show:gr_show(show),inputs=[AS],outputs=[D3])
				def BZ():return sorted(textual_inversion.textual_inversion_templates)
				with gr.Tab(label='Train',id='train'):
					gr.HTML(value='<p style=\'margin-bottom: 0.7em\'>Train an embedding or Hypernetwork; you must specify a directory with a set of 1:1 ratio images <a href="https://github.com/tenebo/standard-demo-we/wiki/Textual-Inversion" style="font-weight:bold;">[wiki]</a></p>')
					with FormRow():AT=gr.Dropdown(label='Embedding',elem_id='train_embedding',choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys()));create_refresh_button(AT,sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings,lambda:{m:sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())},'refresh_train_embedding_name');AU=gr.Dropdown(label='Hypernetwork',elem_id='train_hypernetwork',choices=sorted(shared.hypernetworks));create_refresh_button(AU,shared.reload_hypernetworks,lambda:{m:sorted(shared.hypernetworks)},'refresh_train_hypernetwork_name')
					with FormRow():DC=gr.Textbox(label=Ac,placeholder=Ac,value='0.005',elem_id='train_embedding_learn_rate');DD=gr.Textbox(label=Ab,placeholder=Ab,value='0.00001',elem_id='train_hypernetwork_learn_rate')
					with FormRow():Ba=gr.Dropdown(value=Aa,label='Gradient Clipping',choices=[Aa,'value','norm']);Bb=gr.Textbox(placeholder='Gradient clip value',value='0.1',show_label=_A)
					with FormRow():F=gr.Number(label=N,value=1,precision=0,elem_id='train_batch_size');Bc=gr.Number(label='Gradient accumulation steps',value=1,precision=0,elem_id='train_gradient_step')
					Bd=gr.Textbox(label='Dataset directory',placeholder='Path to directory with input images',elem_id='train_dataset_directory');Be=gr.Textbox(label='Log directory',placeholder='Path to directory where to write outputs',value='textual_inversion',elem_id='train_log_directory')
					with FormRow():AV=gr.Dropdown(label='Prompt template',value='style_filewords.txt',elem_id='train_template_file',choices=BZ());create_refresh_button(AV,textual_inversion.list_textual_inversion_templates,lambda:{m:BZ()},'refrsh_train_template_file')
					Bf=gr.Slider(minimum=64,maximum=2048,step=8,label=p,value=512,elem_id='train_training_width');Bg=gr.Slider(minimum=64,maximum=2048,step=8,label=o,value=512,elem_id='train_training_height');Bh=gr.Checkbox(label='Do not resize images',value=_A,elem_id='train_varsize');D=gr.Number(label='Max steps',value=100000,precision=0,elem_id='train_steps')
					with FormRow():Bi=gr.Number(label='Save an image to log directory every N steps, 0 to disable',value=500,precision=0,elem_id='train_create_image_every');Bj=gr.Number(label='Save a copy of embedding to log directory every N steps, 0 to disable',value=500,precision=0,elem_id='train_save_embedding_every')
					Bk=gr.Checkbox(label='Use PNG alpha channel as loss weight',value=_A,elem_id='use_weight');DE=gr.Checkbox(label='Save images with embedding in PNG chunks',value=_C,elem_id='train_save_image_with_stored_embedding');Bl=gr.Checkbox(label='Read parameters (prompt, etc...) from txt2img tab when making previews',value=_A,elem_id='train_preview_from_txt2img');Bm=gr.Checkbox(label="Shuffle tags by ',' when creating prompts.",value=_A,elem_id='train_shuffle_tags');Bn=gr.Slider(minimum=0,maximum=1,step=.1,label='Drop out tags when creating prompts.',value=0,elem_id='train_tag_drop_out');Bo=gr.Radio(label='Choose latent sampling method',value='once',choices=['once','deterministic','random'],elem_id='train_latent_sampling_method')
					with gr.Row():DF=gr.Button(value='Train Embedding',variant=_G,elem_id='train_train_embedding');DG=gr.Button(value=_N,elem_id='train_interrupt_training');DH=gr.Button(value='Train Hypernetwork',variant=_G,elem_id='train_train_hypernetwork')
				DI=script_callbacks.UiTrainTabParams(AI);script_callbacks.ui_train_tabs_callback(DI)
			with gr.Column(elem_id='ti_gallery_container'):e=gr.Text(elem_id='ti_output',value='',show_label=_A);gr.Gallery(label='Output',show_label=_A,elem_id='ti_gallery',columns=4);gr.HTML(elem_id='ti_progress',value='');f=gr.HTML(elem_id='ti_error',value='')
		Cc.click(fn=textual_inversion_ui.create_embedding,inputs=[CY,CZ,Ca,Cb],outputs=[AT,e,f]);Cm.click(fn=hypernetworks_ui.create_hypernetwork,inputs=[Cd,Ce,Cl,Cf,Cg,Ch,Ci,Cj,Ck],outputs=[AU,e,f]);DB.click(fn=wrap_gradio_gpu_call(textual_inversion_ui.preprocess,extra_outputs=[gr.update()]),_js=z,inputs=[C,Cn,Co,Cp,Cq,Cr,Cs,Ct,AQ,Cu,Cv,Cx,Cy,AR,C_,D0,D1,D2,AS,D4,D5,D6,D7,D8,D9],outputs=[e,f]);DF.click(fn=wrap_gradio_gpu_call(textual_inversion_ui.train_embedding,extra_outputs=[gr.update()]),_js=z,inputs=[C,AT,DC,F,Bc,Bd,Be,Bf,Bg,Bh,D,Ba,Bb,Bm,Bn,Bo,Bk,Bi,Bj,AV,DE,Bl,*AI],outputs=[e,f]);DH.click(fn=wrap_gradio_gpu_call(hypernetworks_ui.train_hypernetwork,extra_outputs=[gr.update()]),_js=z,inputs=[C,AU,DD,F,Bc,Bd,Be,Bf,Bg,Bh,D,Ba,Bb,Bm,Bn,Bo,Bk,Bi,Bj,AV,Bl,*AI],outputs=[e,f]);DG.click(fn=lambda:shared.state.interrupt(),inputs=[],outputs=[]);DA.click(fn=lambda:shared.state.interrupt(),inputs=[],outputs=[])
	S=ui_loadsave.UiLoadsave(cmd_opts.ui_config_file);T=ui_settings.UiSettings();T.create_ui(S,C);g=[(BC,_F,_F),(BK,_D,_D),(CR,'Extras',Ah),(CS,'PNG Info','pnginfo'),(BY.blocks,'Checkpoint Merger','modelmerger'),(CX,'Train','train')];g+=script_callbacks.ui_tabs_callback();g+=[(T.interface,'Settings',AZ)];DJ=ui_extensions.create_ui();g+=[(DJ,'Extensions',AY)];shared.tab_names=[]
	for(DM,w,DN)in g:shared.tab_names.append(w)
	with gr.Blocks(theme=shared.gradio_theme,analytics_enabled=_A,title='Standard Demo')as x:
		T.add_quicksettings();parameters_copypaste.connect_paste_params_buttons()
		with gr.Tabs(elem_id='tabs')as Bp:
			DK={B:A for(A,B)in enumerate(opts.ui_tab_order)};DL=sorted(g,key=lambda x:DK.get(x[1],9999))
			for(Bq,w,y)in DL:
				if w in shared.opts.hidden_tabs:continue
				with gr.TabItem(w,id=y,elem_id=f"tab_{y}"):Bq.render()
				if y not in[AY,AZ]:S.add_block(Bq,y)
			S.add_component(f"webui/Tabs@{Bp.elem_id}",Bp);S.setup_ui()
		if os.path.exists(os.path.join(script_path,AX)):gr.Audio(interactive=_A,value=os.path.join(script_path,AX),elem_id='audio_notification',visible=_A)
		AW=shared.html('footer.html');AW=AW.format(versions=versions_html(),api_docs='/docs'if shared.cmd_opts.api else'https://github.com/tenebo/standard-demo-we/wiki/API');gr.HTML(AW,elem_id='footer');T.add_functionality(x);Br=lambda:gr.update(visible=shared.sd_model and shared.sd_model.cond_stage_key=='edit');T.text_settings.change(fn=Br,inputs=[],outputs=[v]);x.load(fn=Br,inputs=[],outputs=[v]);BY.setup_ui(dummy_component=C,sd_model_checkpoint_component=T.component_dict[_O])
	S.dump_defaults();x.ui_loadsave=S;return x
def versions_html():
	import torch as A,launch as B;D='.'.join([str(A)for A in sys.version_info[0:3]]);E=B.commit_hash();F=B.git_tag()
	if shared.xformers_available:import xformers as G;C=G.__version__
	else:C='N/A'
	return f'''
version: <a href="https://github.com/tenebo/standard-demo-we/commit/{E}">{F}</a>
&#x2000;•&#x2000;
python: <span title="{sys.version}">{D}</span>
&#x2000;•&#x2000;
torch: {getattr(A,"__long_version__",A.__version__)}
&#x2000;•&#x2000;
xformers: {C}
&#x2000;•&#x2000;
gradio: {gr.__version__}
&#x2000;•&#x2000;
checkpoint: <a id="sd_checkpoint_hash">N/A</a>
'''
def setup_ui_api(app):
	A='GET';B=app;from pydantic import BaseModel as F,Field as C;from typing import List
	class D(F):H=C(title='Name of the quicksettings field');I=C(title='Label of the quicksettings field')
	def G():return[D(name=A,label=B.label)for(A,B)in opts.data_labels.items()]
	B.add_api_route('/internal/quicksettings-hint',G,methods=[A],response_model=List[D]);B.add_api_route('/internal/ping',lambda:{},methods=[A]);B.add_api_route('/internal/profile-startup',lambda:timer.startup_record,methods=[A])
	def E(attachment=_A):from fastapi.responses import PlainTextResponse as A;B=sysinfo.get();C=f"sysinfo-{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.txt";return A(B,headers={'Content-Disposition':f'{"attachment"if attachment else"inline"}; filename="{C}"'})
	B.add_api_route('/internal/sysinfo',E,methods=[A]);B.add_api_route('/internal/sysinfo-download',lambda:E(attachment=_C),methods=[A])