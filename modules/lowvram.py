_B='conditioner'
_A=None
import torch
from modules import devices,shared
module_in_gpu=_A
cpu=torch.device('cpu')
def send_everything_to_cpu():
	global module_in_gpu
	if module_in_gpu is not _A:module_in_gpu.to(cpu)
	module_in_gpu=_A
def is_needed(sd_model):return shared.cmd_opts.lowvram or shared.cmd_opts.medvram or shared.cmd_opts.medvram_sdxl and hasattr(sd_model,_B)
def apply(sd_model):
	A=sd_model;B=is_needed(A);shared.parallel_processing_allowed=not B
	if B:setup_for_low_vram(A,not shared.cmd_opts.lowvram)
	else:A.lowvram=False
def setup_for_low_vram(sd_model,use_medvram):
	M='embedder';I='model';A=sd_model
	if getattr(A,'lowvram',False):return
	A.lowvram=True;E={}
	def C(module,_):
		'send this module to GPU; send whatever tracked module was previous in GPU to CPU;\n        we add this as forward_pre_hook to a lot of modules and this way all but one of them will\n        be in CPU\n        ';A=module;global module_in_gpu;A=E.get(A,A)
		if module_in_gpu==A:return
		if module_in_gpu is not _A:module_in_gpu.to(cpu)
		A.to(devices.device);module_in_gpu=A
	N=A.first_stage_model;P=A.first_stage_model.encode;Q=A.first_stage_model.decode
	def R(x):C(N,_A);return P(x)
	def S(z):C(N,_A);return Q(z)
	D=[(A,'first_stage_model'),(A,'depth_model'),(A,M),(A,I),(A,M)];J=hasattr(A,_B);O=not J and hasattr(A.cond_stage_model,I)
	if J:D.append((A,_B))
	elif O:D.append((A.cond_stage_model,I))
	else:D.append((A.cond_stage_model,'transformer'))
	F=[]
	for(G,H)in D:K=getattr(G,H,_A);F.append(K);setattr(G,H,_A)
	A.to(devices.device)
	for((G,H),K)in zip(D,F):setattr(G,H,K)
	if J:A.conditioner.register_forward_pre_hook(C)
	elif O:A.cond_stage_model.model.register_forward_pre_hook(C);A.cond_stage_model.model.token_embedding.register_forward_pre_hook(C);E[A.cond_stage_model.model]=A.cond_stage_model;E[A.cond_stage_model.model.token_embedding]=A.cond_stage_model
	else:A.cond_stage_model.transformer.register_forward_pre_hook(C);E[A.cond_stage_model.transformer]=A.cond_stage_model
	A.first_stage_model.register_forward_pre_hook(C);A.first_stage_model.encode=R;A.first_stage_model.decode=S
	if A.depth_model:A.depth_model.register_forward_pre_hook(C)
	if A.embedder:A.embedder.register_forward_pre_hook(C)
	if use_medvram:A.model.register_forward_pre_hook(C)
	else:
		B=A.model.diffusion_model;F=B.input_blocks,B.middle_block,B.output_blocks,B.time_embed;B.input_blocks,B.middle_block,B.output_blocks,B.time_embed=_A,_A,_A,_A;A.model.to(devices.device);B.input_blocks,B.middle_block,B.output_blocks,B.time_embed=F;B.time_embed.register_forward_pre_hook(C)
		for L in B.input_blocks:L.register_forward_pre_hook(C)
		B.middle_block.register_forward_pre_hook(C)
		for L in B.output_blocks:L.register_forward_pre_hook(C)
def is_enabled(sd_model):return sd_model.lowvram