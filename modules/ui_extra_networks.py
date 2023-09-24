_I='user_metadata'
_H='filename'
_G='metadata'
_F='description'
_E='\\'
_D=True
_C='name'
_B=False
_A=None
import os.path,urllib.parse
from pathlib import Path
from modules import shared,ui_extra_networks_user_metadata,errors,extra_networks
from modules.images import read_info_from_image,save_image_with_geninfo
import gradio as gr,json,html
from fastapi.exceptions import HTTPException
from modules.generation_parameters_copypaste import image_from_url_text
from modules.ui_components import ToolButton
extra_pages=[]
allowed_dirs=set()
def register_page(page):'registers extra networks page for the UI; recommend doing it in on_before_ui() callback for extensions';extra_pages.append(page);allowed_dirs.clear();allowed_dirs.update(set(sum([A.allowed_directories_for_previews()for A in extra_pages],[])))
def fetch_file(filename=''):
	A=filename;from starlette.responses import FileResponse as B
	if not os.path.isfile(A):raise HTTPException(status_code=404,detail='File not found')
	if not any(Path(B).absolute()in Path(A).absolute().parents for B in allowed_dirs):raise ValueError(f"File cannot be fetched: {A}. Must be in one of directories registered by extra pages.")
	C=os.path.splitext(A)[1].lower()
	if C not in('.png','.jpg','.jpeg','.webp','.gif'):raise ValueError(f"File cannot be fetched: {A}. Only png, jpg, webp, and gif.")
	return B(A,headers={'Accept-Ranges':'bytes'})
def get_metadata(page='',item=''):
	A=page;from starlette.responses import JSONResponse as B;A=next(iter([B for B in extra_pages if B.name==A]),_A)
	if A is _A:return B({})
	C=A.metadata.get(item)
	if C is _A:return B({})
	return B({_G:json.dumps(C,indent=4,ensure_ascii=_B)})
def get_single_card(page='',tabname='',name=''):
	C=name;A=page;from starlette.responses import JSONResponse as D;A=next(iter([B for B in extra_pages if B.name==A]),_A)
	try:B=A.create_item(C,enable_filter=_B);A.items[C]=B
	except Exception as E:errors.display(E,'creating item for extra network');B=A.items.get(C)
	A.read_user_metadata(B);F=A.create_html_for_item(B,tabname);return D({'html':F})
def add_pages_to_demo(app):B='GET';A=app;A.add_api_route('/sd_extra_networks/thumb',fetch_file,methods=[B]);A.add_api_route('/sd_extra_networks/metadata',get_metadata,methods=[B]);A.add_api_route('/sd_extra_networks/get-single-card',get_single_card,methods=[B])
def quote_js(s):s=s.replace(_E,'\\\\');s=s.replace('"','\\"');return f'"{s}"'
class ExtraNetworksPage:
	def __init__(A,title):B=title;A.title=B;A.name=B.lower();A.id_page=A.name.replace(' ','_');A.card_page=shared.html('extra-networks-card.html');A.allow_negative_prompt=_B;A.metadata={};A.items={}
	def refresh(A):0
	def read_user_metadata(E,item):
		A=item;D=A.get(_H,_A);B=extra_networks.get_user_metadata(D);C=B.get(_F,_A)
		if C is not _A:A[_F]=C
		A[_I]=B
	def link_preview(D,filename):A=filename;B=urllib.parse.quote(A.replace(_E,'/'));C=os.path.getmtime(A);return f"./sd_extra_networks/thumb?filename={B}&mtime={C}"
	def search_terms_from_path(D,filename,possible_directories=_A):
		B=possible_directories;C=os.path.abspath(filename)
		for A in B if B is not _A else D.allowed_directories_for_previews():
			A=os.path.abspath(A)
			if C.startswith(A):return C[len(A):].replace(_E,'/')
		return''
	def create_html(A,tabname):
		E=tabname;F='';A.metadata={};C={}
		for I in[os.path.abspath(A)for A in A.allowed_directories_for_previews()]:
			for(L,G,Q)in sorted(os.walk(I,followlinks=_D),key=lambda x:shared.natural_sort_key(x[0])):
				for M in sorted(G,key=shared.natural_sort_key):
					H=os.path.join(L,M)
					if not os.path.isdir(H):continue
					B=os.path.abspath(H)[len(I):].replace(_E,'/')
					while B.startswith('/'):B=B[1:]
					N=len(os.listdir(H))==0
					if not N and not B.endswith('/'):B=B+'/'
					if('/.'in B or B.startswith('.'))and not shared.opts.extra_networks_show_hidden_directories:continue
					C[B]=1
		if C:C={'':1,**C}
		O=''.join([f"\n<button class='lg secondary gradio-button custom-button{' search-all'if A==''else''}' onclick='extraNetworksSearchButton(\"{E}_extra_search\", event)'>\n{html.escape(A if A!=''else'all')}\n</button>\n"for A in C]);A.items={A[_C]:A for A in A.list_items()}
		for D in A.items.values():
			J=D.get(_G)
			if J:A.metadata[D[_C]]=J
			if _I not in D:A.read_user_metadata(D)
			F+=A.create_html_for_item(D,E)
		if F=='':G=''.join([f"<li>{A}</li>"for A in A.allowed_directories_for_previews()]);F=shared.html('extra-networks-no-cards.html').format(dirs=G)
		K=A.name.replace(' ','_');P=f"""
<div id='{E}_{K}_subdirs' class='extra-network-subdirs extra-network-subdirs-cards'>
{O}
</div>
<div id='{E}_{K}_cards' class='extra-network-cards'>
{F}
</div>
""";return P
	def create_item(A,name,index=_A):raise NotImplementedError()
	def list_items(A):raise NotImplementedError()
	def allowed_directories_for_previews(A):return[]
	def create_html_for_item(B,item,tabname):
		'\n        Create HTML for card item in tab tabname; can return empty string if the item is not meant to be shown.\n        ';N='search_term';M='sort_keys';H='local_preview';G='prompt';C=tabname;A=item;I=A.get('preview',_A);D=A.get('onclick',_A)
		if D is _A:D='"'+html.escape(f"return cardClicked({quote_js(C)}, {A[G]}, {'true'if B.allow_negative_prompt else'false'})")+'"'
		O=f"height: {shared.opts.extra_networks_card_height}px;"if shared.opts.extra_networks_card_height else'';P=f"width: {shared.opts.extra_networks_card_width}px;"if shared.opts.extra_networks_card_width else'';Q=f'<img src="{html.escape(I)}" class="preview" loading="lazy">'if I else'';J='';R=A.get(_G)
		if R:J=f"<div class='metadata-button card-button' title='Show internal metadata' onclick='extraNetworksRequestMetadata(event, {quote_js(B.name)}, {quote_js(A[_C])})'></div>"
		S=f"<div class='edit-button card-button' title='Edit metadata' onclick='extraNetworksEditUserMetadata(event, {quote_js(C)}, {quote_js(B.id_page)}, {quote_js(A[_C])})'></div>";E='';K=A.get(_H,'')
		for T in B.allowed_directories_for_previews():
			L=os.path.abspath(T)
			if K.startswith(L):E=K[len(L):]
		if shared.opts.extra_networks_hidden_models=='Always':F=_B
		else:F='/.'in E or'\\.'in E
		if F and shared.opts.extra_networks_hidden_models=='Never':return''
		U=' '.join([html.escape(f"data-sort-{A}={B}")for(A,B)in A.get(M,{}).items()]).strip();V={'background_image':Q,'style':f"'display: none; {O}{P}; font-size: {shared.opts.extra_networks_card_text_scale*100}%'",G:A.get(G,_A),'tabname':quote_js(C),H:quote_js(A[H]),_C:html.escape(A[_C]),_F:A.get(_F)or''if shared.opts.extra_networks_card_show_desc else'','card_clicked':D,'save_card_preview':'"'+html.escape(f"return saveCardPreview(event, {quote_js(C)}, {quote_js(A[H])})")+'"',N:A.get(N,''),'metadata_button':J,'edit_button':S,'search_only':' search_only'if F else'',M:U};return B.card_page.format(**V)
	def get_sort_keys(C,path):'\n        List of default keys used for sorting in the UI.\n        ';A=Path(path);B=A.stat();return{'date_created':int(B.st_ctime or 0),'date_modified':int(B.st_mtime or 0),_C:A.name.lower()}
	def find_preview(C,path):
		'\n        Find a preview PNG for a given path (without extension) and call link_preview on it.\n        ';A=['png','jpg','jpeg','webp']
		if shared.opts.samples_format not in A:A.append(shared.opts.samples_format)
		D=sum([[path+'.'+A,path+'.preview.'+A]for A in A],[])
		for B in D:
			if os.path.isfile(B):return C.link_preview(B)
	def find_description(C,path):
		'\n        Find and read a description file for a given path (without extension).\n        '
		for A in[f"{path}.txt",f"{path}.description.txt"]:
			try:
				with open(A,'r',encoding='utf-8',errors='replace')as B:return B.read()
			except OSError:pass
	def create_user_metadata_editor(A,ui,tabname):return ui_extra_networks_user_metadata.UserMetadataEditor(ui,tabname,A)
def initialize():extra_pages.clear()
def register_default_pages():from modules.ui_extra_networks_textual_inversion import ExtraNetworksPageTextualInversion as A;from modules.ui_extra_networks_hypernets import ExtraNetworksPageHypernetworks as B;from modules.ui_extra_networks_checkpoints import ExtraNetworksPageCheckpoints as C;register_page(A());register_page(B());register_page(C())
class ExtraNetworksUi:
	def __init__(A):A.pages=_A;"gradio HTML components related to extra networks' pages";A.page_contents=_A;'HTML content of the above; empty initially, filled when extra pages have to be shown';A.stored_extra_pages=_A;A.button_save_preview=_A;A.preview_target_filename=_A;A.tabname=_A
def pages_in_preferred_order(pages):
	A=pages;C=[A.lower().strip()for A in shared.opts.ui_extra_networks_tab_reorder.split(',')]
	def B(name):
		B=name;B=B.lower()
		for(D,E)in enumerate(C):
			if E in B:return D
		return len(A)
	D={A.name:(B(A.name),C)for(C,A)in enumerate(A)};return sorted(A,key=lambda x:D[x.name])
def create_ui(interface,unrelated_tabs,tabname):
	N='Default Sort';B=tabname;from modules.ui import switch_values_symbol as O;A=ExtraNetworksUi();A.pages=[];A.pages_contents=[];A.user_metadata_editors=[];A.stored_extra_pages=pages_in_preferred_order(extra_pages.copy());A.tabname=B;F=[]
	for D in A.stored_extra_pages:
		with gr.Tab(D.title,id=D.id_page)as C:P=f"{B}_{D.id_page}_cards_html";G=gr.HTML('Loading...',elem_id=P);A.pages.append(G);G.change(fn=lambda:_A,_js='function(){applyExtraNetworkFilter('+quote_js(B)+'); return []}',inputs=[],outputs=[]);H=D.create_user_metadata_editor(A,B);H.create_ui();A.user_metadata_editors.append(H);F.append(C)
	I=gr.Textbox('',show_label=_B,elem_id=B+'_extra_search',elem_classes='search',placeholder='Search...',visible=_B,interactive=_D);J=gr.Dropdown(choices=[N,'Date Created','Date Modified','Name'],value=N,elem_id=B+'_extra_sort',elem_classes='sort',multiselect=_B,visible=_B,show_label=_B,interactive=_D,label=B+'_extra_sort_order');K=ToolButton(O,elem_id=B+'_extra_sortorder',elem_classes='sortorder',visible=_B);E=gr.Button('Refresh',elem_id=B+'_extra_refresh',visible=_B);L=gr.Checkbox(_D,label='Show dirs',elem_id=B+'_extra_show_dirs',elem_classes='show-dirs',visible=_B);A.button_save_preview=gr.Button('Save preview',elem_id=B+'_save_preview',visible=_B);A.preview_target_filename=gr.Textbox('Preview save filename',elem_id=B+'_preview_filename',visible=_B)
	for C in unrelated_tabs:C.select(fn=lambda:[gr.update(visible=_B)for A in range(5)],inputs=[],outputs=[I,J,K,E,L],show_progress=_B)
	for C in F:C.select(fn=lambda:[gr.update(visible=_D)for A in range(5)],inputs=[],outputs=[I,J,K,E,L],show_progress=_B)
	def Q():
		if not A.pages_contents:return M()
		return A.pages_contents
	def M():
		for B in A.stored_extra_pages:B.refresh()
		A.pages_contents=[B.create_html(A.tabname)for B in A.stored_extra_pages];return A.pages_contents
	interface.load(fn=Q,inputs=[],outputs=[*A.pages]);E.click(fn=M,inputs=[],outputs=A.pages);return A
def path_is_parent(parent_path,child_path):B=child_path;A=parent_path;A=os.path.abspath(A);B=os.path.abspath(B);return B.startswith(A)
def setup_ui(ui,gallery):
	B=gallery;A=ui
	def C(index,images,filename):
		D=filename;C=images;B=index
		if len(C)==0:print('There is no image in gallery to save as a preview.');return[B.create_html(A.tabname)for B in A.stored_extra_pages]
		B=int(B);B=0 if B<0 else B;B=len(C)-1 if B>=len(C)else B;G=C[B if B>=0 else 0];E=image_from_url_text(G);H,J=read_info_from_image(E);F=_B
		for I in A.stored_extra_pages:
			if any(path_is_parent(A,D)for A in I.allowed_directories_for_previews()):F=_D;break
		assert F,f"writing to {D} is not allowed";save_image_with_geninfo(E,H,D);return[B.create_html(A.tabname)for B in A.stored_extra_pages]
	A.button_save_preview.click(fn=C,_js='function(x, y, z){return [selected_gallery_index(), y, z]}',inputs=[A.preview_target_filename,B,A.preview_target_filename],outputs=[*A.pages])
	for D in A.user_metadata_editors:D.setup_ui(B)