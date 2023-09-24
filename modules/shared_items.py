_B='None'
_A='Automatic'
import sys
from modules.shared_cmd_options import cmd_opts
def realesrgan_models_names():import modules.realesrgan_model;return[A.name for A in modules.realesrgan_model.get_realesrgan_models(None)]
def postprocessing_scripts():import modules.scripts;return modules.scripts.scripts_postproc.scripts
def sd_vae_items():import modules.sd_vae;return[_A,_B]+list(modules.sd_vae.vae_dict)
def refresh_vae_list():import modules.sd_vae;modules.sd_vae.refresh_vae_list()
def cross_attention_optimizations():import modules.sd_hijack;return[_A]+[A.title()for A in modules.sd_hijack.optimizers]+[_B]
def sd_unet_items():import modules.sd_unet;return[_A]+[A.label for A in modules.sd_unet.unet_options]+[_B]
def refresh_unet_list():import modules.sd_unet;modules.sd_unet.list_unets()
def list_checkpoint_tiles():import modules.sd_models;return modules.sd_models.checkpoint_tiles()
def refresh_checkpoints():import modules.sd_models;return modules.sd_models.list_models()
def list_samplers():import modules.sd_samplers;return modules.sd_samplers.all_samplers
def reload_hypernetworks():from modules.hypernetworks import hypernetwork as A;from modules import shared as B;B.hypernetworks=A.list_hypernetworks(cmd_opts.hypernetwork_dir)
ui_reorder_categories_builtin_items=['inpaint','sampler','accordions','checkboxes','dimensions','cfg','denoising','seed','batch','override_settings']
def ui_reorder_categories():
	from modules import scripts as B;yield from ui_reorder_categories_builtin_items;C={}
	for A in B.scripts_txt2img.scripts+B.scripts_img2img.scripts:
		if isinstance(A.section,str)and A.section not in ui_reorder_categories_builtin_items:C[A.section]=1
	yield from C;yield'scripts'
class Shared(sys.modules[__name__].__class__):
	'\n    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than\n    at program startup.\n    ';sd_model_val=None
	@property
	def sd_model(self):import modules.sd_models;return modules.sd_models.model_data.get_sd_model()
	@sd_model.setter
	def sd_model(self,value):import modules.sd_models;modules.sd_models.model_data.set_sd_model(value)
sys.modules['modules.shared'].__class__=Shared