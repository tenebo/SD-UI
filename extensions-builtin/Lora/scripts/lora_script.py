_C='Alias from file'
_B='lora_in_memory_limit'
_A='choices'
import re,gradio as gr
from fastapi import FastAPI
import network,networks,lora,lora_patches,extra_networks_lora,ui_extra_networks_lora
from modules import script_callbacks,ui_extra_networks,extra_networks,shared
def unload():networks.originals.undo()
def before_ui():ui_extra_networks.register_page(ui_extra_networks_lora.ExtraNetworksPageLora());networks.extra_network_lora=extra_networks_lora.ExtraNetworkLora();extra_networks.register_extra_network(networks.extra_network_lora);extra_networks.register_extra_network_alias(networks.extra_network_lora,'lyco')
networks.originals=lora_patches.LoraPatches()
script_callbacks.on_model_loaded(networks.assign_network_names_to_compvis_modules)
script_callbacks.on_script_unloaded(unload)
script_callbacks.on_before_ui(before_ui)
script_callbacks.on_infotext_pasted(networks.infotext_pasted)
shared.options_templates.update(shared.options_section(('extra_networks','Extra Networks'),{'sd_lora':shared.OptionInfo('None','Add network to prompt',gr.Dropdown,lambda:{_A:['None',*networks.available_networks]},refresh=networks.list_available_networks),'lora_preferred_name':shared.OptionInfo(_C,'When adding to prompt, refer to Lora by',gr.Radio,{_A:[_C,'Filename']}),'lora_add_hashes_to_infotext':shared.OptionInfo(True,'Add Lora hashes to infotext'),'lora_show_all':shared.OptionInfo(False,'Always show all networks on the Lora page').info('otherwise, those detected as for incompatible version of Stable Diffusion will be hidden'),'lora_hide_unknown_for_versions':shared.OptionInfo([],'Hide networks of unknown versions for model versions',gr.CheckboxGroup,{_A:['SD1','SD2','SDXL']}),_B:shared.OptionInfo(0,'Number of Lora networks to keep cached in memory',gr.Number,{'precision':0})}))
shared.options_templates.update(shared.options_section(('compatibility','Compatibility'),{'lora_functional':shared.OptionInfo(False,'Lora/Networks: use old method that takes longer when you have multiple Loras active and produces same results as kohya-ss/sd-ourui-additional-networks extension')}))
def create_lora_json(obj):A=obj;return{'name':A.name,'alias':A.alias,'path':A.filename,'metadata':A.metadata}
def api_networks(_,app):
	@app.get('/sdapi/v1/loras')
	async def A():return[create_lora_json(A)for A in networks.available_networks.values()]
	@app.post('/sdapi/v1/refresh-loras')
	async def B():return networks.list_available_networks()
script_callbacks.on_app_started(api_networks)
re_lora=re.compile('<lora:([^:]+):')
def infotext_pasted(infotext,d):
	B='Prompt';A=d.get('Lora hashes')
	if not A:return
	A=[A.strip().split(':',1)for A in A.split(',')];A={A[0].strip().replace(',',''):A[1].strip()for A in A}
	def C(m):
		D=m.group(1);B=A.get(D)
		if B is None:return m.group(0)
		C=networks.available_network_hash_lookup.get(B)
		if C is None:return m.group(0)
		return f"<lora:{C.get_alias()}:"
	d[B]=re.sub(re_lora,C,d[B])
script_callbacks.on_infotext_pasted(infotext_pasted)
shared.opts.onchange(_B,networks.purge_networks_from_memory)