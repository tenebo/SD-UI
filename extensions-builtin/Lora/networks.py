_F='network_layer_name'
_E='network_bias_backup'
_D='network_weights_backup'
_C='attentions'
_B=True
_A=None
import logging,os,re,lora_patches,network,network_lora,network_hada,network_ia3,network_lokr,network_full,network_norm,torch
from typing import Union
from modules import shared,devices,sd_models,errors,scripts,sd_hijack
module_types=[network_lora.ModuleTypeLora(),network_hada.ModuleTypeHada(),network_ia3.ModuleTypeIa3(),network_lokr.ModuleTypeLokr(),network_full.ModuleTypeFull(),network_norm.ModuleTypeNorm()]
re_digits=re.compile('\\d+')
re_x_proj=re.compile('(.*)_([qkv]_proj)$')
re_compiled={}
suffix_conversion={_C:{},'resnets':{'conv1':'in_layers_2','conv2':'out_layers_3','norm1':'in_layers_0','norm2':'out_layers_0','time_emb_proj':'emb_layers_1','conv_shortcut':'skip_connection'}}
def convert_diffusers_name_to_compvis(key,is_sd2):
	F='attn';G='self_attn';H='mlp_c_proj';I='mlp_c_fc';D='mlp_fc2';E='mlp_fc1'
	def B(match_list,regex_text):
		C=match_list;B=regex_text;A=re_compiled.get(B)
		if A is _A:A=re.compile(B);re_compiled[B]=A
		D=re.match(A,key)
		if not D:return False
		C.clear();C.extend([int(A)if re.match(re_digits,A)else A for A in D.groups()]);return _B
	A=[]
	if B(A,'lora_unet_conv_in(.*)'):return f"diffusion_model_input_blocks_0_0{A[0]}"
	if B(A,'lora_unet_conv_out(.*)'):return f"diffusion_model_out_2{A[0]}"
	if B(A,'lora_unet_time_embedding_linear_(\\d+)(.*)'):return f"diffusion_model_time_embed_{A[0]*2-2}{A[1]}"
	if B(A,'lora_unet_down_blocks_(\\d+)_(attentions|resnets)_(\\d+)_(.+)'):C=suffix_conversion.get(A[1],{}).get(A[3],A[3]);return f"diffusion_model_input_blocks_{1+A[0]*3+A[2]}_{1 if A[1]==_C else 0}_{C}"
	if B(A,'lora_unet_mid_block_(attentions|resnets)_(\\d+)_(.+)'):C=suffix_conversion.get(A[0],{}).get(A[2],A[2]);return f"diffusion_model_middle_block_{1 if A[0]==_C else A[1]*2}_{C}"
	if B(A,'lora_unet_up_blocks_(\\d+)_(attentions|resnets)_(\\d+)_(.+)'):C=suffix_conversion.get(A[1],{}).get(A[3],A[3]);return f"diffusion_model_output_blocks_{A[0]*3+A[2]}_{1 if A[1]==_C else 0}_{C}"
	if B(A,'lora_unet_down_blocks_(\\d+)_downsamplers_0_conv'):return f"diffusion_model_input_blocks_{3+A[0]*3}_0_op"
	if B(A,'lora_unet_up_blocks_(\\d+)_upsamplers_0_conv'):return f"diffusion_model_output_blocks_{2+A[0]*3}_{2 if A[0]>0 else 1}_conv"
	if B(A,'lora_te_text_model_encoder_layers_(\\d+)_(.+)'):
		if is_sd2:
			if E in A[1]:return f"model_transformer_resblocks_{A[0]}_{A[1].replace(E,I)}"
			elif D in A[1]:return f"model_transformer_resblocks_{A[0]}_{A[1].replace(D,H)}"
			else:return f"model_transformer_resblocks_{A[0]}_{A[1].replace(G,F)}"
		return f"transformer_text_model_encoder_layers_{A[0]}_{A[1]}"
	if B(A,'lora_te2_text_model_encoder_layers_(\\d+)_(.+)'):
		if E in A[1]:return f"1_model_transformer_resblocks_{A[0]}_{A[1].replace(E,I)}"
		elif D in A[1]:return f"1_model_transformer_resblocks_{A[0]}_{A[1].replace(D,H)}"
		else:return f"1_model_transformer_resblocks_{A[0]}_{A[1].replace(G,F)}"
	return key
def assign_network_names_to_compvis_modules(sd_model):
	E='_';D={}
	if shared.sd_model.is_sdxl:
		for(G,F)in enumerate(shared.sd_model.conditioner.embedders):
			if not hasattr(F,'wrapped'):continue
			for(C,A)in F.wrapped.named_modules():B=f"{G}_{C.replace('.',E)}";D[B]=A;A.network_layer_name=B
	else:
		for(C,A)in shared.sd_model.cond_stage_model.wrapped.named_modules():B=C.replace('.',E);D[B]=A;A.network_layer_name=B
	for(C,A)in shared.sd_model.model.named_modules():B=C.replace('.',E);D[B]=A;A.network_layer_name=B
	sd_model.network_layer_mapping=D
def load_network(name,network_on_disk):
	K='lora_unet';H='lora_te1_text_model';D=network_on_disk;E=network.Network(name,D);E.mtime=os.path.getmtime(D.filename);N=sd_models.read_state_dict(D.filename)
	if not hasattr(shared.sd_model,'network_layer_mapping'):assign_network_names_to_compvis_modules(shared.sd_model)
	I={};O='model_transformer_resblocks'in shared.sd_model.network_layer_mapping;F={}
	for(J,P)in N.items():
		C,Q=J.split('.',1);A=convert_diffusers_name_to_compvis(C,O);B=shared.sd_model.network_layer_mapping.get(A,_A)
		if B is _A:
			L=re_x_proj.match(A)
			if L:B=shared.sd_model.network_layer_mapping.get(L.group(1),_A)
		if B is _A and K in C:A=C.replace(K,'diffusion_model');B=shared.sd_model.network_layer_mapping.get(A,_A)
		elif B is _A and H in C:
			A=C.replace(H,'0_transformer_text_model');B=shared.sd_model.network_layer_mapping.get(A,_A)
			if B is _A:A=C.replace(H,'transformer_text_model');B=shared.sd_model.network_layer_mapping.get(A,_A)
		if B is _A:I[J]=A;continue
		if A not in F:F[A]=network.NetworkWeights(network_key=J,sd_key=A,w={},sd_module=B)
		F[A].w[Q]=P
	for(A,M)in F.items():
		G=_A
		for R in module_types:
			G=R.create_module(E,M)
			if G is not _A:break
		if G is _A:raise AssertionError(f"Could not find a module type (out of {', '.join([A.__class__.__name__ for A in module_types])}) that would accept those keys: {', '.join(M.w)}")
		E.modules[A]=G
	if I:logging.debug(f"Network {D.filename} didn't match keys: {I}")
	return E
def purge_networks_from_memory():
	while len(networks_in_memory)>shared.opts.lora_in_memory_limit and len(networks_in_memory)>0:A=next(iter(networks_in_memory));networks_in_memory.pop(A,_A)
	devices.torch_gc()
def load_networks(names,te_multipliers=_A,unet_multipliers=_A,dyn_dims=_A):
	I=dyn_dims;J=unet_multipliers;K=te_multipliers;E=1.;D=names;L={}
	for A in loaded_networks:
		if A.name in D:L[A.name]=A
	loaded_networks.clear();F=[available_network_aliases.get(A,_A)for A in D]
	if any(A is _A for A in F):list_available_networks();F=[available_network_aliases.get(A,_A)for A in D]
	G=[]
	for(H,(C,B))in enumerate(zip(F,D)):
		A=L.get(B,_A)
		if C is not _A:
			if A is _A:A=networks_in_memory.get(B)
			if A is _A or os.path.getmtime(C.filename)>A.mtime:
				try:A=load_network(B,C);networks_in_memory.pop(B,_A);networks_in_memory[B]=A
				except Exception as M:errors.display(M,f"loading network {C.filename}");continue
			A.mentioned_name=B;C.read_hash()
		if A is _A:G.append(B);logging.info(f"Couldn't find network with name {B}");continue
		A.te_multiplier=K[H]if K else E;A.unet_multiplier=J[H]if J else E;A.dyn_dim=I[H]if I else E;loaded_networks.append(A)
	if G:sd_hijack.model_hijack.comments.append('Networks not found: '+', '.join(G))
	purge_networks_from_memory()
def network_restore_weights_from_backup(self):
	A=self;B=getattr(A,_D,_A);C=getattr(A,_E,_A)
	if B is _A and C is _A:return
	if B is not _A:
		if isinstance(A,torch.nn.MultiheadAttention):A.in_proj_weight.copy_(B[0]);A.out_proj.weight.copy_(B[1])
		else:A.weight.copy_(B)
	if C is not _A:
		if isinstance(A,torch.nn.MultiheadAttention):A.out_proj.bias.copy_(C)
		else:A.bias.copy_(C)
	elif isinstance(A,torch.nn.MultiheadAttention):A.out_proj.bias=_A
	else:A.bias=_A
def network_apply_weights(self):
	'\n    Applies the currently selected set of networks to the weights of torch layer self.\n    If weights already have this particular set of networks applied, does nothing.\n    If not, restores orginal weights from backup and alters weights according to networks.\n    ';K='bias';A=self;C=getattr(A,_F,_A)
	if C is _A:return
	L=getattr(A,'network_current_names',());G=tuple((A.name,A.te_multiplier,A.unet_multiplier,A.dyn_dim)for A in loaded_networks);F=getattr(A,_D,_A)
	if F is _A and G!=():
		if L!=():raise RuntimeError('no backup weights found and current weights are not unchanged')
		if isinstance(A,torch.nn.MultiheadAttention):F=A.in_proj_weight.to(devices.cpu,copy=_B),A.out_proj.weight.to(devices.cpu,copy=_B)
		else:F=A.weight.to(devices.cpu,copy=_B)
		A.network_weights_backup=F
	E=getattr(A,_E,_A)
	if E is _A:
		if isinstance(A,torch.nn.MultiheadAttention)and A.out_proj.bias is not _A:E=A.out_proj.bias.to(devices.cpu,copy=_B)
		elif getattr(A,K,_A)is not _A:E=A.bias.to(devices.cpu,copy=_B)
		else:E=_A
		A.network_bias_backup=E
	if L!=G:
		network_restore_weights_from_backup(A)
		for B in loaded_networks:
			H=B.modules.get(C,_A)
			if H is not _A and hasattr(A,'weight'):
				try:
					with torch.no_grad():
						I,D=H.calc_updown(A.weight)
						if len(A.weight.shape)==4 and A.weight.shape[1]==9:I=torch.nn.functional.pad(I,(0,0,0,0,0,5))
						A.weight+=I
						if D is not _A and hasattr(A,K):
							if A.bias is _A:A.bias=torch.nn.Parameter(D)
							else:A.bias+=D
				except RuntimeError as J:logging.debug(f"Network {B.name} layer {C}: {J}");extra_network_lora.errors[B.name]=extra_network_lora.errors.get(B.name,0)+1
				continue
			M=B.modules.get(C+'_q_proj',_A);N=B.modules.get(C+'_k_proj',_A);O=B.modules.get(C+'_v_proj',_A);P=B.modules.get(C+'_out_proj',_A)
			if isinstance(A,torch.nn.MultiheadAttention)and M and N and O and P:
				try:
					with torch.no_grad():R,Q=M.calc_updown(A.in_proj_weight);S,Q=N.calc_updown(A.in_proj_weight);T,Q=O.calc_updown(A.in_proj_weight);U=torch.vstack([R,S,T]);V,D=P.calc_updown(A.out_proj.weight);A.in_proj_weight+=U;A.out_proj.weight+=V
					if D is not _A:
						if A.out_proj.bias is _A:A.out_proj.bias=torch.nn.Parameter(D)
						else:A.out_proj.bias+=D
				except RuntimeError as J:logging.debug(f"Network {B.name} layer {C}: {J}");extra_network_lora.errors[B.name]=extra_network_lora.errors.get(B.name,0)+1
				continue
			if H is _A:continue
			logging.debug(f"Network {B.name} layer {C}: couldn't find supported operation");extra_network_lora.errors[B.name]=extra_network_lora.errors.get(B.name,0)+1
		A.network_current_names=G
def network_forward(module,input,original_forward):
	"\n    Old way of applying Lora by executing operations during layer's forward.\n    Stacking many loras this way results in big performance degradation.\n    ";C=original_forward;A=module
	if len(loaded_networks)==0:return C(A,input)
	input=devices.cond_cast_unet(input);network_restore_weights_from_backup(A);network_reset_cached_weight(A);B=C(A,input);D=getattr(A,_F,_A)
	for E in loaded_networks:
		A=E.modules.get(D,_A)
		if A is _A:continue
		B=A.forward(input,B)
	return B
def network_reset_cached_weight(self):self.network_current_names=();self.network_weights_backup=_A
def network_Linear_forward(self,input):
	A=self
	if shared.opts.lora_functional:return network_forward(A,input,originals.Linear_forward)
	network_apply_weights(A);return originals.Linear_forward(A,input)
def network_Linear_load_state_dict(self,*A,**B):network_reset_cached_weight(self);return originals.Linear_load_state_dict(self,*A,**B)
def network_Conv2d_forward(self,input):
	A=self
	if shared.opts.lora_functional:return network_forward(A,input,originals.Conv2d_forward)
	network_apply_weights(A);return originals.Conv2d_forward(A,input)
def network_Conv2d_load_state_dict(self,*A,**B):network_reset_cached_weight(self);return originals.Conv2d_load_state_dict(self,*A,**B)
def network_GroupNorm_forward(self,input):
	A=self
	if shared.opts.lora_functional:return network_forward(A,input,originals.GroupNorm_forward)
	network_apply_weights(A);return originals.GroupNorm_forward(A,input)
def network_GroupNorm_load_state_dict(self,*A,**B):network_reset_cached_weight(self);return originals.GroupNorm_load_state_dict(self,*A,**B)
def network_LayerNorm_forward(self,input):
	A=self
	if shared.opts.lora_functional:return network_forward(A,input,originals.LayerNorm_forward)
	network_apply_weights(A);return originals.LayerNorm_forward(A,input)
def network_LayerNorm_load_state_dict(self,*A,**B):network_reset_cached_weight(self);return originals.LayerNorm_load_state_dict(self,*A,**B)
def network_MultiheadAttention_forward(self,*A,**B):network_apply_weights(self);return originals.MultiheadAttention_forward(self,*A,**B)
def network_MultiheadAttention_load_state_dict(self,*A,**B):network_reset_cached_weight(self);return originals.MultiheadAttention_load_state_dict(self,*A,**B)
def list_available_networks():
	D='.safetensors';E='.ckpt';F='.pt';available_networks.clear();available_network_aliases.clear();forbidden_network_aliases.clear();available_network_hash_lookup.clear();forbidden_network_aliases.update({'none':1,'Addams':1});os.makedirs(shared.cmd_opts.lora_dir,exist_ok=_B);G=list(shared.walk_files(shared.cmd_opts.lora_dir,allowed_extensions=[F,E,D]));G+=list(shared.walk_files(shared.cmd_opts.lyco_dir_backcompat,allowed_extensions=[F,E,D]))
	for B in G:
		if os.path.isdir(B):continue
		C=os.path.splitext(os.path.basename(B))[0]
		try:A=network.NetworkOnDisk(C,B)
		except OSError:errors.report(f"Failed to load network {C} from {B}",exc_info=_B);continue
		available_networks[C]=A
		if A.alias in available_network_aliases:forbidden_network_aliases[A.alias.lower()]=1
		available_network_aliases[C]=A;available_network_aliases[A.alias]=A
re_network_name=re.compile('(.*)\\s*\\([0-9a-fA-F]+\\)')
def infotext_pasted(infotext,params):
	E='AddNet Model ';A=params
	if'AddNet Module 1'in[A[1]for A in scripts.scripts_txt2img.infotext_fields]:return
	C=[]
	for F in A:
		if not F.startswith(E):continue
		D=F[13:]
		if A.get('AddNet Module '+D)!='LoRA':continue
		B=A.get(E+D)
		if B is _A:continue
		G=re_network_name.match(B)
		if G:B=G.group(1)
		H=A.get('AddNet Weight A '+D,'1.0');C.append(f"<lora:{B}:{H}>")
	if C:A['Prompt']+='\n'+''.join(C)
originals=_A
extra_network_lora=_A
available_networks={}
available_network_aliases={}
loaded_networks=[]
networks_in_memory={}
available_network_hash_lookup={}
forbidden_network_aliases={}
list_available_networks()