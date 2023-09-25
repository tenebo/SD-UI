_H='preferred weight'
_G='activation text'
_F='Unknown'
_E='metadata'
_D=', '
_C=False
_B=True
_A=None
import datetime,html,random,gradio as gr,re
from modules import ui_extra_networks_user_metadata
def is_non_comma_tagset(tags):A=sum(len(A)for A in tags.keys())/len(tags);return A>=16
re_word=re.compile("[-_\\w']+")
re_comma=re.compile(' *, *')
def build_tags(metadata):
	A={}
	for(J,E)in metadata.get('ss_tag_frequency',{}).items():
		for(B,F)in E.items():B=B.strip();A[B]=A.get(B,0)+int(F)
	if A and is_non_comma_tagset(A):
		C={}
		for(G,H)in A.items():
			for D in re.findall(re_word,G):
				if len(D)<3:continue
				C[D]=C.get(D,0)+H
		A=C
	I=sorted(A.keys(),key=A.get,reverse=_B);return[(B,A[B])for B in I]
class LoraUserMetadataEditor(ui_extra_networks_user_metadata.UserMetadataEditor):
	def __init__(A,ui,tabname,page):super().__init__(ui,tabname,page);A.select_sd_version=_A;A.taginfo=_A;A.edit_activation_text=_A;A.slider_preferred_weight=_A;A.edit_notes=_A
	def save_lora_user_metadata(B,name,desc,sd_version,activation_text,preferred_weight,notes):A=B.get_user_metadata(name);A['description']=desc;A['sd version']=sd_version;A[_G]=activation_text;A[_H]=preferred_weight;A['notes']=notes;B.write_user_metadata(name,A)
	def get_metadata_table(M,name):
		L='buckets';A=super().get_metadata_table(name);N=M.page.items.get(name,{});D=N.get(_E)or{};O={'ss_output_name':'Output name:','ss_sd_model_name':'Model:','ss_clip_skip':'Clip skip:','ss_network_module':'Kohya module:'}
		for(P,Q)in O.items():
			F=D.get(P,_A)
			if F is not _A and str(F)!='None':A.append((Q,html.escape(F)))
		J=D.get('ss_training_started_at')
		if J:A.append(('Date trained:',datetime.datetime.utcfromtimestamp(float(J)).strftime('%Y-%m-%d %H:%M')))
		G=D.get('ss_bucket_info')
		if G and L in G:
			B={}
			for(R,K)in G[L].items():C=K['resolution'];C=f"{C[1]}x{C[0]}";B[C]=B.get(C,0)+int(K['count'])
			H=sorted(B.keys(),key=B.get,reverse=_B);E=html.escape(_D.join(H[0:4]))
			if len(B)>4:E+=', ...';E=f"<span title='{html.escape(_D.join(H))}'>{E}</span>"
			A.append(('Resolutions:'if len(H)>1 else'Resolution:',E))
		I=0
		for(R,S)in D.get('ss_dataset_dirs',{}).items():I+=int(S.get('img_count',0))
		if I:A.append(('Dataset size:',I))
		return A
	def put_values_into_components(B,name):C=name;D=B.get_user_metadata(C);F=super().put_values_into_components(C);E=B.page.items.get(C,{});G=E.get(_E)or{};A=build_tags(G);H=[(A,str(B))for(A,B)in A[0:24]];return[*F[0:5],E.get('sd_version',_F),gr.HighlightedText.update(value=H,visible=_B if A else _C),D.get(_G,''),float(D.get(_H,.0)),gr.update(visible=_B if A else _C),gr.update(value=B.generate_random_prompt_from_tags(A),visible=_B if A else _C)]
	def generate_random_prompt(A,name):B=A.page.items.get(name,{});C=B.get(_E)or{};D=build_tags(C);return A.generate_random_prompt_from_tags(D)
	def generate_random_prompt_from_tags(F,tags):
		A=_A;B=[]
		for(D,C)in tags:
			if not A:A=C
			E=random.random()*A
			if C>E:B.append(D)
		return _D.join(sorted(B))
	def create_extra_default_items_in_left_column(A):A.select_sd_version=gr.Dropdown(['SD1','SD2','SDXL',_F],value=_F,label='Stable Diffusion version',interactive=_B)
	def create_editor(A):
		A.create_default_editor_elems();A.taginfo=gr.HighlightedText(label='Training dataset tags');A.edit_activation_text=gr.Text(label='Activation text',info='Will be added to prompt along with Lora');A.slider_preferred_weight=gr.Slider(label='Preferred weight',info='Set to 0 to disable',minimum=.0,maximum=2.,step=.01)
		with gr.Row()as C:
			with gr.Column(scale=8):B=gr.Textbox(label='Random prompt',lines=4,max_lines=4,interactive=_C)
			with gr.Column(scale=1,min_width=120):D=gr.Button('Generate',size='lg',scale=1)
		A.edit_notes=gr.TextArea(label='Notes',lines=4);D.click(fn=A.generate_random_prompt,inputs=[A.edit_name_input],outputs=[B],show_progress=_C)
		def E(activation_text,evt):
			C=activation_text;A=evt.value[0];B=re.split(re_comma,C)
			if A in B:B=[B for B in B if B!=A and B.strip()];return _D.join(B)
			return C+_D+A if C else A
		A.taginfo.select(fn=E,inputs=[A.edit_activation_text],outputs=[A.edit_activation_text],show_progress=_C);A.create_default_buttons();F=[A.edit_name,A.edit_description,A.html_filedata,A.html_preview,A.edit_notes,A.select_sd_version,A.taginfo,A.edit_activation_text,A.slider_preferred_weight,C,B];A.button_edit.click(fn=A.put_values_into_components,inputs=[A.edit_name_input],outputs=F).then(fn=lambda:gr.update(visible=_B),inputs=[],outputs=[A.box]);G=[A.edit_description,A.select_sd_version,A.edit_activation_text,A.slider_preferred_weight,A.edit_notes];A.setup_save_handler(A.button_save,A.save_lora_user_metadata,G)