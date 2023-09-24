_A=None
import ldm.modules.encoders.modules,open_clip,torch,transformers.utils.hub
from modules import shared
class ReplaceHelper:
	def __init__(A):A.replaced=[]
	def replace(D,obj,field,func):
		A=field;B=obj;C=getattr(B,A,_A)
		if C is _A:return
		D.replaced.append((B,A,C));setattr(B,A,func);return C
	def restore(A):
		for(B,C,D)in A.replaced:setattr(B,C,D)
		A.replaced.clear()
class DisableInitialization(ReplaceHelper):
	"\n    When an object of this class enters a `with` block, it starts:\n    - preventing torch's layer initialization functions from working\n    - changes CLIP and OpenCLIP to not download model weights\n    - changes CLIP to not make requests to check if there is a new version of a file you already have\n\n    When it leaves the block, it reverts everything to how it was before.\n\n    Use it like this:\n    ```\n    with DisableInitialization():\n        do_things()\n    ```\n    "
	def __init__(A,disable_clip=True):super().__init__();A.disable_clip=disable_clip
	def replace(D,obj,field,func):
		A=field;B=obj;C=getattr(B,A,_A)
		if C is _A:return
		D.replaced.append((B,A,C));setattr(B,A,func);return C
	def __enter__(A):
		E='cached_file';B=False
		def C(*A,**B):0
		def F(*B,pretrained=_A,**C):return A.create_model_and_transforms(*B,pretrained=_A,**C)
		def G(pretrained_model_name_or_path,*D,**E):B=pretrained_model_name_or_path;C=A.CLIPTextModel_from_pretrained(_A,*D,config=B,state_dict={},**E);C.name_or_path=B;return C
		def H(*B,**C):B=B[0:3]+('/',)+B[4:];return A.transformers_modeling_utils_load_pretrained_model(*B,**C)
		def D(original,url,*C,**D):
			E=original;A=url
			if A=='https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/added_tokens.json'or A=='openai/clip-vit-large-patch14'and C[0]=='added_tokens.json':return
			try:
				F=E(A,*C,local_files_only=True,**D)
				if F is _A:F=E(A,*C,local_files_only=B,**D)
				return F
			except Exception:return E(A,*C,local_files_only=B,**D)
		def I(url,*B,local_files_only=B,**C):return D(A.transformers_utils_hub_get_from_cache,url,*B,**C)
		def J(url,*B,local_files_only=B,**C):return D(A.transformers_tokenization_utils_base_cached_file,url,*B,**C)
		def K(url,*B,local_files_only=B,**C):return D(A.transformers_configuration_utils_cached_file,url,*B,**C)
		A.replace(torch.nn.init,'kaiming_uniform_',C);A.replace(torch.nn.init,'_no_grad_normal_',C);A.replace(torch.nn.init,'_no_grad_uniform_',C)
		if A.disable_clip:A.create_model_and_transforms=A.replace(open_clip,'create_model_and_transforms',F);A.CLIPTextModel_from_pretrained=A.replace(ldm.modules.encoders.modules.CLIPTextModel,'from_pretrained',G);A.transformers_modeling_utils_load_pretrained_model=A.replace(transformers.modeling_utils.PreTrainedModel,'_load_pretrained_model',H);A.transformers_tokenization_utils_base_cached_file=A.replace(transformers.tokenization_utils_base,E,J);A.transformers_configuration_utils_cached_file=A.replace(transformers.configuration_utils,E,K);A.transformers_utils_hub_get_from_cache=A.replace(transformers.utils.hub,'get_from_cache',I)
	def __exit__(A,exc_type,exc_val,exc_tb):A.restore()
class InitializeOnMeta(ReplaceHelper):
	'\n    Context manager that causes all parameters for linear/conv2d/mha layers to be allocated on meta device,\n    which results in those parameters having no values and taking no memory. model.to() will be broken and\n    will need to be repaired by using LoadStateDictOnMeta below when loading params from state dict.\n\n    Usage:\n    ```\n    with sd_disable_initialization.InitializeOnMeta():\n        sd_model = instantiate_from_config(sd_config.model)\n    ```\n    '
	def __enter__(A):
		B='__init__'
		if shared.cmd_opts.disable_model_loading_ram_optimization:return
		def C(x):x['device']='meta';return x
		D=A.replace(torch.nn.Linear,B,lambda*A,**B:D(*A,**C(B)));E=A.replace(torch.nn.Conv2d,B,lambda*A,**B:E(*A,**C(B)));F=A.replace(torch.nn.MultiheadAttention,B,lambda*A,**B:F(*A,**C(B)));A.replace(torch.nn.Module,'to',lambda*A,**B:_A)
	def __exit__(A,exc_type,exc_val,exc_tb):A.restore()
class LoadStateDictOnMeta(ReplaceHelper):
	'\n    Context manager that allows to read parameters from state_dict into a model that has some of its parameters in the meta device.\n    As those parameters are read from state_dict, they will be deleted from it, so by the end state_dict will be mostly empty, to save memory.\n    Meant to be used together with InitializeOnMeta above.\n\n    Usage:\n    ```\n    with sd_disable_initialization.LoadStateDictOnMeta(state_dict):\n        model.load_state_dict(state_dict, strict=False)\n    ```\n    '
	def __init__(A,state_dict,device,weight_dtype_conversion=_A):super().__init__();A.state_dict=state_dict;A.device=device;A.weight_dtype_conversion=weight_dtype_conversion or{};A.default_dtype=A.weight_dtype_conversion.get('')
	def get_weight_dtype(A,key):B,C=key.split('.',1);return A.weight_dtype_conversion.get(B,A.default_dtype)
	def __enter__(A):
		B='_load_from_state_dict'
		if shared.cmd_opts.disable_model_loading_ram_optimization:return
		H=A.state_dict;K=A.device
		def C(original,module,state_dict,prefix,*L,**M):
			I=prefix;E=state_dict;F=module;J=[]
			for(G,D)in F._parameters.items():
				if D is _A:continue
				B=I+G;C=H.pop(B,_A)
				if C is not _A:E[B]=C.to(dtype=A.get_weight_dtype(B));J.append(B)
				if D.is_meta:N=C.dtype if C is not _A else D.dtype;F._parameters[G]=torch.nn.parameter.Parameter(torch.zeros_like(D,device=K,dtype=N),requires_grad=D.requires_grad)
			for G in F._buffers:
				B=I+G;C=H.pop(B,_A)
				if C is not _A:E[B]=C;J.append(B)
			original(F,E,I,*L,**M)
			for B in J:E.pop(B,_A)
		def D(original,module,state_dict,strict=True):
			"torch makes a lot of copies of the dictionary with weights, so just deleting entries from state_dict does not help\n            because the same values are stored in multiple copies of the dict. The trick used here is to give torch a dict with\n            all weights on meta device, i.e. deleted, and then it doesn't matter how many copies torch makes.\n\n            In _load_from_state_dict, the correct weight will be obtained from a single dict with the right weights (sd).\n\n            The dangerous thing about this is if _load_from_state_dict is not called, (if some exotic module overloads\n            the function and does not call the original) the state dict will just fail to load because weights\n            would be on the meta device.\n            ";A=state_dict
			if A==H:A={B:A.to(device='meta',dtype=A.dtype)for(B,A)in A.items()}
			original(module,A,strict=strict)
		E=A.replace(torch.nn.Module,'load_state_dict',lambda*A,**B:D(E,*A,**B));F=A.replace(torch.nn.Module,B,lambda*A,**B:C(F,*A,**B));G=A.replace(torch.nn.Linear,B,lambda*A,**B:C(G,*A,**B));I=A.replace(torch.nn.Conv2d,B,lambda*A,**B:C(I,*A,**B));J=A.replace(torch.nn.MultiheadAttention,B,lambda*A,**B:C(J,*A,**B));L=A.replace(torch.nn.LayerNorm,B,lambda*A,**B:C(L,*A,**B));M=A.replace(torch.nn.GroupNorm,B,lambda*A,**B:C(M,*A,**B))
	def __exit__(A,exc_type,exc_val,exc_tb):A.restore()