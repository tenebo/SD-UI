from modules import extra_networks,shared
from modules.hypernetworks import hypernetwork
class ExtraNetworkHypernet(extra_networks.ExtraNetwork):
	def __init__(A):super().__init__('hypernet')
	def activate(G,p,params_list):
		C=params_list;A=shared.opts.sd_hypernetwork
		if A!='None'and A in shared.hypernetworks and not any(B for B in C if B.items[0]==A):F=f"<hypernet:{A}:{shared.opts.extra_networks_default_multiplier}>";p.all_prompts=[f"{A}{F}"for A in p.all_prompts];C.append(extra_networks.ExtraNetworkParams(items=[A,shared.opts.extra_networks_default_multiplier]))
		D=[];E=[]
		for B in C:assert B.items;D.append(B.items[0]);E.append(float(B.items[1])if len(B.items)>1 else 1.)
		hypernetwork.load_hypernetworks(D,E)
	def deactivate(A,p):0