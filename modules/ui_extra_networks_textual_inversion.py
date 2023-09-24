import os
from modules import ui_extra_networks,sd_hijack,shared
from modules.ui_extra_networks import quote_js
class ExtraNetworksPageTextualInversion(ui_extra_networks.ExtraNetworksPage):
	def __init__(A):super().__init__('Textual Inversion');A.allow_negative_prompt=True
	def refresh(A):sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)
	def create_item(B,name,index=None,enable_filter=True):A=sd_hijack.model_hijack.embedding_db.word_embeddings.get(name);C,D=os.path.splitext(A.filename);return{'name':name,'filename':A.filename,'shorthash':A.shorthash,'preview':B.find_preview(C),'description':B.find_description(C),'search_term':B.search_terms_from_path(A.filename)+' '+(A.hash or''),'prompt':quote_js(A.name),'local_preview':f"{C}.preview.{shared.opts.samples_format}",'sort_keys':{'default':index,**B.get_sort_keys(A.filename)}}
	def list_items(A):
		for(B,C)in enumerate(sd_hijack.model_hijack.embedding_db.word_embeddings):yield A.create_item(C,B)
	def allowed_directories_for_previews(A):return list(sd_hijack.model_hijack.embedding_db.embedding_dirs)