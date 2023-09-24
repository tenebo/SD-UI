_A=None
import html,gradio as gr,modules.hypernetworks.hypernetwork
from modules import devices,sd_hijack,shared
not_available=['hardswish','multiheadattention']
keys=[A for A in modules.hypernetworks.hypernetwork.HypernetworkModule.activation_dict if A not in not_available]
def create_hypernetwork(name,enable_sizes,overwrite_old,layer_structure=_A,activation_func=_A,weight_init=_A,add_layer_norm=False,use_dropout=False,dropout_structure=_A):A=modules.hypernetworks.hypernetwork.create_hypernetwork(name,enable_sizes,overwrite_old,layer_structure,activation_func,weight_init,add_layer_norm,use_dropout,dropout_structure);return gr.Dropdown.update(choices=sorted(shared.hypernetworks)),f"Created: {A}",''
def train_hypernetwork(*A):
	shared.loaded_hypernetworks=[];assert not shared.cmd_opts.lowvram,'Training models with lowvram is not possible'
	try:sd_hijack.undo_optimizations();B,C=modules.hypernetworks.hypernetwork.train_hypernetwork(*A);D=f"\nTraining {'interrupted'if shared.state.interrupted else'finished'} at {B.step} steps.\nHypernetwork saved to {html.escape(C)}\n";return D,''
	except Exception:raise
	finally:shared.sd_model.cond_stage_model.to(devices.device);shared.sd_model.first_stage_model.to(devices.device);sd_hijack.apply_optimizations()