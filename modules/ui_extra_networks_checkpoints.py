import html,os
from modules import shared,ui_extra_networks,sd_models
from modules.ui_extra_networks import quote_js
from modules.ui_extra_networks_checkpoints_user_metadata import CheckpointUserMetadataEditor
class ExtraNetworksPageCheckpoints(ui_extra_networks.ExtraNetworksPage):
	def __init__(A):super().__init__('Checkpoints')
	def refresh(A):shared.refresh_checkpoints()
	def create_item(B,name,index=None,enable_filter=True):A=sd_models.checkpoint_aliases.get(name);C,D=os.path.splitext(A.filename);return{'name':A.name_for_extra,'filename':A.filename,'shorthash':A.shorthash,'preview':B.find_preview(C),'description':B.find_description(C),'search_term':B.search_terms_from_path(A.filename)+' '+(A.sha256 or''),'onclick':'"'+html.escape(f"return selectCheckpoint({quote_js(name)})")+'"','local_preview':f"{C}.{shared.opts.samples_format}",'metadata':A.metadata,'sort_keys':{'default':index,**B.get_sort_keys(A.filename)}}
	def list_items(A):
		B=list(sd_models.checkpoints_list)
		for(C,D)in enumerate(B):yield A.create_item(D,C)
	def allowed_directories_for_previews(A):return[A for A in[shared.cmd_opts.ckpt_dir,sd_models.model_path]if A is not None]
	def create_user_metadata_editor(A,ui,tabname):return CheckpointUserMetadataEditor(ui,tabname,A)