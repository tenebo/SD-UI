from modules import sd_samplers_kdiffusion,sd_samplers_timesteps,shared
from modules.sd_samplers_common import samples_to_image_grid,sample_to_image
all_samplers=[*sd_samplers_kdiffusion.samplers_data_k_diffusion,*sd_samplers_timesteps.samplers_data_timesteps]
all_samplers_map={A.name:A for A in all_samplers}
samplers=[]
samplers_for_img2img=[]
samplers_map={}
samplers_hidden={}
def find_sampler_config(name):
	if name is not None:A=all_samplers_map.get(name,None)
	else:A=all_samplers[0]
	return A
def create_sampler(name,model):
	B=model;A=find_sampler_config(name);assert A is not None,f"bad sampler name: {name}"
	if B.is_sdxl and A.options.get('no_sdxl',False):raise Exception(f"Sampler {A.name} is not supported for SDXL")
	C=A.constructor(B);C.config=A;return C
def set_samplers():
	global samplers,samplers_for_img2img,samplers_hidden;samplers_hidden=set(shared.opts.hide_samplers);samplers=all_samplers;samplers_for_img2img=all_samplers;samplers_map.clear()
	for A in all_samplers:
		samplers_map[A.name.lower()]=A.name
		for B in A.aliases:samplers_map[B.lower()]=A.name
def visible_sampler_names():return[A.name for A in samplers if A.name not in samplers_hidden]
set_samplers()