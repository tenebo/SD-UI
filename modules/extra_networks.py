_A=None
import json,os,re,logging
from collections import defaultdict
from modules import errors
extra_network_registry={}
extra_network_aliases={}
def initialize():extra_network_registry.clear();extra_network_aliases.clear()
def register_extra_network(extra_network):A=extra_network;extra_network_registry[A.name]=A
def register_extra_network_alias(extra_network,alias):extra_network_aliases[alias]=extra_network
def register_default_extra_networks():from modules.extra_networks_hypernet import ExtraNetworkHypernet as A;register_extra_network(A())
class ExtraNetworkParams:
	def __init__(A,items=_A):
		A.items=items or[];A.positional=[];A.named={}
		for B in A.items:
			C=B.split('=',2)if isinstance(B,str)else[B]
			if len(C)==2:A.named[C[0]]=C[1]
			else:A.positional.append(B)
	def __eq__(A,other):return A.items==other.items
class ExtraNetwork:
	def __init__(A,name):A.name=name
	def activate(A,p,params_list):'\n        Called by processing on every run. Whatever the extra network is meant to do should be activated here.\n        Passes arguments related to this extra network in params_list.\n        User passes arguments by specifying this in his prompt:\n\n        <name:arg1:arg2:arg3>\n\n        Where name matches the name of this ExtraNetwork object, and arg1:arg2:arg3 are any natural number of text arguments\n        separated by colon.\n\n        Even if the user does not mention this ExtraNetwork in his prompt, the call will stil be made, with empty params_list -\n        in this case, all effects of this extra networks should be disabled.\n\n        Can be called multiple times before deactivate() - each new call should override the previous call completely.\n\n        For example, if this ExtraNetwork\'s name is \'hypernet\' and user\'s prompt is:\n\n        > "1girl, <hypernet:agm:1.1> <extrasupernet:master:12:13:14> <hypernet:ray>"\n\n        params_list will be:\n\n        [\n            ExtraNetworkParams(items=["agm", "1.1"]),\n            ExtraNetworkParams(items=["ray"])\n        ]\n\n        ';raise NotImplementedError
	def deactivate(A,p):'\n        Called at the end of processing for housekeeping. No need to do anything here.\n        ';raise NotImplementedError
def lookup_extra_networks(extra_network_data):
	"returns a dict mapping ExtraNetwork objects to lists of arguments for those extra networks.\n\n    Example input:\n    {\n        'lora': [<modules.extra_networks.ExtraNetworkParams object at 0x0000020690D58310>],\n        'lyco': [<modules.extra_networks.ExtraNetworkParams object at 0x0000020690D58F70>],\n        'hypernet': [<modules.extra_networks.ExtraNetworkParams object at 0x0000020690D5A800>]\n    }\n\n    Example output:\n\n    {\n        <extra_networks_lora.ExtraNetworkLora object at 0x0000020581BEECE0>: [<modules.extra_networks.ExtraNetworkParams object at 0x0000020690D58310>, <modules.extra_networks.ExtraNetworkParams object at 0x0000020690D58F70>],\n        <modules.extra_networks_hypernet.ExtraNetworkHypernet object at 0x0000020581BEEE60>: [<modules.extra_networks.ExtraNetworkParams object at 0x0000020690D5A800>]\n    }\n    ";C={}
	for(B,E)in list(extra_network_data.items()):
		A=extra_network_registry.get(B,_A);D=extra_network_aliases.get(B,_A)
		if D is not _A and A is _A:A=D
		if A is _A:logging.info(f"Skipping unknown extra network: {B}");continue
		C.setdefault(A,[]).extend(E)
	return C
def activate(p,extra_network_data):
	'call activate for extra networks in extra_network_data in specified order, then call\n    activate for all remaining registered networks with an empty argument list';C=extra_network_data;D=[]
	for(A,E)in lookup_extra_networks(C).items():
		try:A.activate(p,E);D.append(A)
		except Exception as B:errors.display(B,f"activating extra network {A.name} with arguments {E}")
	for(F,A)in extra_network_registry.items():
		if A in D:continue
		try:A.activate(p,[])
		except Exception as B:errors.display(B,f"activating extra network {F}")
	if p.scripts is not _A:p.scripts.after_extra_networks_activate(p,batch_number=p.iteration,prompts=p.prompts,seeds=p.seeds,subseeds=p.subseeds,extra_network_data=C)
def deactivate(p,extra_network_data):
	'call deactivate for extra networks in extra_network_data in specified order, then call\n    deactivate for all remaining registered networks';C=lookup_extra_networks(extra_network_data)
	for A in C:
		try:A.deactivate(p)
		except Exception as B:errors.display(B,f"deactivating extra network {A.name}")
	for(D,A)in extra_network_registry.items():
		if A in C:continue
		try:A.deactivate(p)
		except Exception as B:errors.display(B,f"deactivating unmentioned extra network {D}")
re_extra_net=re.compile('<(\\w+):([^>]+)>')
def parse_prompt(prompt):
	A=prompt;B=defaultdict(list)
	def C(m):A=m.group(1);C=m.group(2);B[A].append(ExtraNetworkParams(items=C.split(':')));return''
	A=re.sub(re_extra_net,C,A);return A,B
def parse_prompts(prompts):
	B=[];A=_A
	for C in prompts:
		D,E=parse_prompt(C)
		if A is _A:A=E
		B.append(D)
	return B,A
def get_user_metadata(filename):
	B=filename
	if B is _A:return{}
	D,G=os.path.splitext(B);A=D+'.json';C={}
	try:
		if os.path.isfile(A):
			with open(A,'r',encoding='utf8')as E:C=json.load(E)
	except Exception as F:errors.display(F,f"reading extra network user metadata from {A}")
	return C