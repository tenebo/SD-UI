_D=', name);}'
_C='filename'
_B='description'
_A=None
import datetime,html,json,os.path,gradio as gr
from modules import generation_parameters_copypaste,images,sysinfo,errors,ui_extra_networks
class UserMetadataEditor:
	def __init__(A,ui,tabname,page):A.ui=ui;A.tabname=tabname;A.page=page;A.id_part=f"{A.tabname}_{A.page.id_page}_edit_user_metadata";A.box=_A;A.edit_name_input=_A;A.button_edit=_A;A.edit_name=_A;A.edit_description=_A;A.edit_notes=_A;A.html_filedata=_A;A.html_preview=_A;A.html_status=_A;A.button_cancel=_A;A.button_replace_preview=_A;A.button_save=_A
	def get_user_metadata(D,name):
		C='user_metadata';B=D.page.items.get(name,{});A=B.get(C,_A)
		if not A:A={_B:B.get(_B,'')};B[C]=A
		return A
	def create_extra_default_items_in_left_column(A):0
	def create_default_editor_elems(A):
		with gr.Row():
			with gr.Column(scale=2):A.edit_name=gr.HTML(elem_classes='extra-network-name');A.edit_description=gr.Textbox(label='Description',lines=4);A.html_filedata=gr.HTML();A.create_extra_default_items_in_left_column()
			with gr.Column(scale=1,min_width=0):A.html_preview=gr.HTML()
	def create_default_buttons(A):
		B='primary'
		with gr.Row(elem_classes='edit-user-metadata-buttons'):A.button_cancel=gr.Button('Cancel');A.button_replace_preview=gr.Button('Replace preview',variant=B);A.button_save=gr.Button('Save',variant=B)
		A.html_status=gr.HTML(elem_classes='edit-user-metadata-status');A.button_cancel.click(fn=_A,_js='closePopup')
	def get_card_html(C,name):
		E='preview';B=C.page.items.get(name,{});A=B.get(E,_A)
		if not A:F,G=os.path.splitext(B[_C]);A=C.page.find_preview(F);B[E]=A
		if A:D=f'\n            <div class=\'card standalone-card-preview\'>\n                <img src="{html.escape(A)}" class="preview">\n            </div>\n            '
		else:D="<div class='card standalone-card-preview'></div>"
		return D
	def relative_path(C,path):
		A=path
		for B in C.page.allowed_directories_for_previews():
			if ui_extra_networks.path_is_parent(B,A):return os.path.relpath(A,B)
		return os.path.basename(A)
	def get_metadata_table(A,name):
		B=A.page.items.get(name,{})
		try:C=B[_C];E=B.get('shorthash',_A);D=os.stat(C);F=[('Filename: ',A.relative_path(C)),('File size: ',sysinfo.pretty_bytes(D.st_size)),('Hash: ',E),('Modified: ',datetime.datetime.fromtimestamp(D.st_mtime).strftime('%Y-%m-%d %H:%M'))];return F
		except Exception as G:errors.display(G,f"reading info for {name}");return[]
	def put_values_into_components(B,name):
		A=name;C=B.get_user_metadata(A)
		try:D=B.get_metadata_table(A)
		except Exception as E:errors.display(E,f"reading metadata info for {A}");D=[]
		F='<table class="file-metadata">'+''.join(f"<tr><th>{B}</th><td>{A}</td></tr>"for(B,A)in D if A is not _A)+'</table>';return html.escape(A),C.get(_B,''),F,B.get_card_html(A),C.get('notes','')
	def write_user_metadata(A,name,metadata):
		B=A.page.items.get(name,{});C=B.get(_C,_A);D,F=os.path.splitext(C)
		with open(D+'.json','w',encoding='utf8')as E:json.dump(metadata,E,indent=4)
	def save_user_metadata(B,name,desc,notes):A=B.get_user_metadata(name);A[_B]=desc;A['notes']=notes;B.write_user_metadata(name,A)
	def setup_save_handler(A,button,func,components):button.click(fn=func,inputs=[A.edit_name_input,*components],outputs=[]).then(fn=_A,_js='function(name){closePopup(); extraNetworksRefreshSingleCard('+json.dumps(A.page.name)+','+json.dumps(A.tabname)+_D,inputs=[A.edit_name_input],outputs=[])
	def create_editor(A):A.create_default_editor_elems();A.edit_notes=gr.TextArea(label='Notes',lines=4);A.create_default_buttons();A.button_edit.click(fn=A.put_values_into_components,inputs=[A.edit_name_input],outputs=[A.edit_name,A.edit_description,A.html_filedata,A.html_preview,A.edit_notes]).then(fn=lambda:gr.update(visible=True),inputs=[],outputs=[A.box]);A.setup_save_handler(A.button_save,A.save_user_metadata,[A.edit_description,A.edit_notes])
	def create_ui(A):
		B=False
		with gr.Box(visible=B,elem_id=A.id_part,elem_classes='edit-user-metadata')as C:A.box=C;A.edit_name_input=gr.Textbox('Edit user metadata card id',visible=B,elem_id=f"{A.id_part}_name");A.button_edit=gr.Button('Edit user metadata',visible=B,elem_id=f"{A.id_part}_button");A.create_editor()
	def save_preview(C,index,gallery,name):
		D=name;B=gallery;A=index
		if len(B)==0:return C.get_card_html(D),'There is no image in gallery to save as a preview.'
		F=C.page.items.get(D,{});A=int(A);A=0 if A<0 else A;A=len(B)-1 if A>=len(B)else A;G=B[A if A>=0 else 0];E=generation_parameters_copypaste.image_from_url_text(G);H,I=images.read_info_from_image(E);images.save_image_with_geninfo(E,H,F['local_preview']);return C.get_card_html(D),''
	def setup_ui(A,gallery):A.button_replace_preview.click(fn=A.save_preview,_js='function(x, y, z){return [selected_gallery_index(), y, z]}',inputs=[A.edit_name_input,gallery,A.edit_name_input],outputs=[A.html_preview,A.html_status]).then(fn=_A,_js='function(name){extraNetworksRefreshSingleCard('+json.dumps(A.page.name)+','+json.dumps(A.tabname)+_D,inputs=[A.edit_name_input],outputs=[])