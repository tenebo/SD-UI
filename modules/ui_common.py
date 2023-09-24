_C=True
_B=False
_A=None
import json,html,os,platform,sys,gradio as gr,subprocess as sp
from modules import call_queue,shared
from modules.generation_parameters_copypaste import image_from_url_text
import modules.images
from modules.ui_components import ToolButton
import modules.generation_parameters_copypaste as parameters_copypaste
folder_symbol='📂'
refresh_symbol='🔄'
def update_generation_info(generation_info,html_info,img_index):
	C='infotexts';D=html_info;B=img_index;A=generation_info
	try:
		A=json.loads(A)
		if B<0 or B>=len(A[C]):return D,gr.update()
		return plaintext_to_html(A[C][B]),gr.update()
	except Exception:pass
	return D,gr.update()
def plaintext_to_html(text,classname=_A):A=classname;B='<br>\n'.join(html.escape(A)for A in text.split('\n'));return f"<p class='{A}'>{B}</p>"if A else f"<p>{B}</p>"
def save_files(js_data,images,do_make_zip,index):
	K='negative_prompt';L='steps';M='height';N='width';O='seed';P='prompt';H=images;D=index;import csv;E=[];C=[]
	class Y:
		def __init__(A,d=_A):
			if d is not _A:
				for(B,C)in d.items():setattr(A,B,C)
	B=json.loads(js_data);A=Y(B);I=shared.opts.outdir_save;Z=shared.opts.use_save_to_dirs_for_ui;a=shared.opts.samples_format;Q=0;R=_B
	if D>-1 and shared.opts.save_selected_only and D>=B['index_of_first_image']:R=_C;H=[H[D]];Q=D
	os.makedirs(shared.opts.outdir_save,exist_ok=_C)
	with open(os.path.join(shared.opts.outdir_save,'log.csv'),'a',encoding='utf8',newline='')as S:
		b=S.tell()==0;T=csv.writer(S)
		if b:T.writerow([P,O,N,M,'sampler','cfgs',L,'filename',K])
		for(G,c)in enumerate(H,Q):
			U=image_from_url_text(c);V=G<A.index_of_first_image;F=0 if V else G-A.index_of_first_image;A.batch_index=G-1;W,J=modules.images.save_image(U,I,'',seed=A.all_seeds[F],prompt=A.all_prompts[F],extension=a,info=A.infotexts[G],grid=V,p=A,save_to_dirs=Z);d=os.path.relpath(W,I);E.append(d);C.append(W)
			if J:E.append(os.path.basename(J));C.append(J)
		T.writerow([B[P],B[O],B[N],B[M],B['sampler_name'],B['cfg_scale'],B[L],E[0],B[K]])
	if do_make_zip:
		e=A.all_seeds[D-1]if R else A.all_seeds[0];f=modules.images.FilenameGenerator(A,e,A.all_prompts[0],U,_C);g=f.apply(shared.opts.grid_zip_filename_pattern or'[datetime]_[[model_name]]_[seed]-[seed_last]');X=os.path.join(I,f"{g}.zip");from zipfile import ZipFile as h
		with h(X,'w')as i:
			for F in range(len(C)):
				with open(C[F],mode='rb')as j:i.writestr(E[F],j.read())
		C.insert(0,X)
	return gr.File.update(value=C,visible=_C),plaintext_to_html(f"Saved: {E[0]}")
def create_output_panel(tabname,outdir):
	J='infotext';H='img2img';D='txt2img';E='extras';A=tabname
	def L(f):
		if not os.path.exists(f):print(f'Folder "{f}" does not exist. After you create an image, the folder will be created.');return
		elif not os.path.isdir(f):print(f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
""",file=sys.stderr);return
		if not shared.cmd_opts.hide_ui_dir_config:
			A=os.path.normpath(f)
			if platform.system()=='Windows':os.startfile(A)
			elif platform.system()=='Darwin':sp.Popen(['open',A])
			elif'microsoft-standard-WSL2'in platform.uname().release:sp.Popen(['wsl-open',A])
			else:sp.Popen(['xdg-open',A])
	with gr.Column(variant='panel',elem_id=f"{A}_results"):
		with gr.Group(elem_id=f"{A}_gallery_container"):F=gr.Gallery(label='Output',show_label=_B,elem_id=f"{A}_gallery",columns=4,preview=_C,height=shared.opts.gallery_height or _A)
		C=_A
		with gr.Column():
			with gr.Row(elem_id=f"image_buttons_{A}",elem_classes='image-buttons'):
				M=ToolButton(folder_symbol,elem_id=f"{A}_open_folder",visible=not shared.cmd_opts.hide_ui_dir_config,tooltip='Open images output directory.')
				if A!=E:N=ToolButton('💾',elem_id=f"save_{A}",tooltip=f"Save the image to a dedicated directory ({shared.opts.outdir_save}).");O=ToolButton('🗃️',elem_id=f"save_zip_{A}",tooltip=f"Save zip archive with images to a dedicated directory ({shared.opts.outdir_save})")
				P={H:ToolButton('🖼️',elem_id=f"{A}_send_to_img2img",tooltip='Send image and generation parameters to img2img tab.'),'inpaint':ToolButton('🎨️',elem_id=f"{A}_send_to_inpaint",tooltip='Send image and generation parameters to img2img inpaint tab.'),E:ToolButton('📐',elem_id=f"{A}_send_to_extras",tooltip='Send image and generation parameters to extras tab.')}
			M.click(fn=lambda:L(shared.opts.outdir_samples or outdir),inputs=[],outputs=[])
			if A!=E:
				K=gr.File(_A,file_count='multiple',interactive=_B,show_label=_B,visible=_B,elem_id=f"download_files_{A}")
				with gr.Group():
					B=gr.HTML(elem_id=f"html_info_{A}",elem_classes=J);G=gr.HTML(elem_id=f"html_log_{A}",elem_classes='html-log');C=gr.Textbox(visible=_B,elem_id=f"generation_info_{A}")
					if A==D or A==H:Q=gr.Button(visible=_B,elem_id=f"{A}_generation_info_button");Q.click(fn=update_generation_info,_js='function(x, y, z){ return [x, y, selected_gallery_index()] }',inputs=[C,B,B],outputs=[B,B],show_progress=_B)
					N.click(fn=call_queue.wrap_gradio_call(save_files),_js='(x, y, z, w) => [x, y, false, selected_gallery_index()]',inputs=[C,F,B,B],outputs=[K,G],show_progress=_B);O.click(fn=call_queue.wrap_gradio_call(save_files),_js='(x, y, z, w) => [x, y, true, selected_gallery_index()]',inputs=[C,F,B,B],outputs=[K,G])
			else:R=gr.HTML(elem_id=f"html_info_x_{A}");B=gr.HTML(elem_id=f"html_info_{A}",elem_classes=J);G=gr.HTML(elem_id=f"html_log_{A}")
			I=[]
			if A==D:I=modules.scripts.scripts_txt2img.paste_field_names
			elif A==H:I=modules.scripts.scripts_img2img.paste_field_names
			for(S,T)in P.items():parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=T,tabname=S,source_tabname=D if A==D else _A,source_image_component=F,paste_field_names=I))
			return F,C if A!=E else R,B,G
def create_refresh_button(refresh_component,refresh_method,refreshed_args,elem_id):
	C=refreshed_args;D=refresh_component;A=D if isinstance(D,list)else[D];B=_A
	for F in A:
		B=getattr(F,'label',_A)
		if B is not _A:break
	def G():
		refresh_method();B=C()if callable(C)else C
		for(D,E)in B.items():
			for F in A:setattr(F,D,E)
		return[gr.update(**B or{})for A in A]if len(A)>1 else gr.update(**B or{})
	E=ToolButton(value=refresh_symbol,elem_id=elem_id,tooltip=f"{B}: refresh"if B else'Refresh');E.click(fn=G,inputs=[],outputs=A);return E
def setup_dialog(button_show,dialog,*,button_close=_A):
	'Sets up the UI so that the dialog (gr.Box) is invisible, and is only shown when buttons_show is clicked, in a fullscreen modal window.';B=button_close;A=dialog;A.visible=_B;button_show.click(fn=lambda:gr.update(visible=_C),inputs=[],outputs=[A]).then(fn=_A,_js="function(){ popupId('"+A.elem_id+"'); }")
	if B:B.click(fn=_A,_js='closePopup')