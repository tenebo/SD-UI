import os
from modules import shared,ui_extra_networks
from modules.ui_extra_networks import quote_js
from modules.hashes import sha256_from_cache
class ExtraNetworksPageHypernetworks(ui_extra_networks.ExtraNetworksPage):
	def __init__(A):super().__init__('Hypernetworks')
	def refresh(A):shared.reload_hypernetworks()
	def create_item(B,name,index=None,enable_filter=True):C=name;D=shared.hypernetworks[C];A,F=os.path.splitext(D);E=sha256_from_cache(D,f"hypernet/{C}");G=E[0:10]if E else None;return{'name':C,'filename':D,'shorthash':G,'preview':B.find_preview(A),'description':B.find_description(A),'search_term':B.search_terms_from_path(A)+' '+(E or''),'prompt':quote_js(f"<hypernet:{C}:")+' + opts.extra_networks_default_multiplier + '+quote_js('>'),'local_preview':f"{A}.preview.{shared.opts.samples_format}",'sort_keys':{'default':index,**B.get_sort_keys(A+F)}}
	def list_items(A):
		for(B,C)in enumerate(shared.hypernetworks):yield A.create_item(C,B)
	def allowed_directories_for_previews(A):return[shared.cmd_opts.hypernetwork_dir]