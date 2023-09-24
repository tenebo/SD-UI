_D='circular'
_C='conditioner'
_B=False
_A=None
import torch
from torch.nn.functional import silu
from types import MethodType
from modules import devices,sd_hijack_optimizations,shared,script_callbacks,errors,sd_unet
from modules.hypernetworks import hypernetwork
from modules.shared import cmd_opts
from modules import sd_hijack_clip,sd_hijack_open_clip,sd_hijack_unet,sd_hijack_xlmr,xlmr
import ldm.modules.attention,ldm.modules.diffusionmodules.model,ldm.modules.diffusionmodules.openaimodel,ldm.models.diffusion.ddim,ldm.models.diffusion.plms,ldm.modules.encoders.modules,sgm.modules.attention,sgm.modules.diffusionmodules.model,sgm.modules.diffusionmodules.openaimodel,sgm.modules.encoders.modules
attention_CrossAttention_forward=ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity=ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward=ldm.modules.diffusionmodules.model.AttnBlock.forward
ldm.modules.attention.MemoryEfficientCrossAttention=ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES['softmax-xformers']=ldm.modules.attention.CrossAttention
ldm.modules.attention.print=shared.ldm_print
ldm.modules.diffusionmodules.model.print=shared.ldm_print
ldm.util.print=shared.ldm_print
ldm.models.diffusion.ddpm.print=shared.ldm_print
optimizers=[]
current_optimizer=_A
def list_optimizers():A=script_callbacks.list_optimizers_callback();A=[A for A in A if A.is_available()];A=sorted(A,key=lambda x:x.priority,reverse=True);optimizers.clear();optimizers.extend(A)
def apply_optimizations(option=_A):
	C='Automatic';global current_optimizer;undo_optimizations()
	if len(optimizers)==0:current_optimizer=_A;return''
	ldm.modules.diffusionmodules.model.nonlinearity=silu;ldm.modules.diffusionmodules.openaimodel.th=sd_hijack_unet.th;sgm.modules.diffusionmodules.model.nonlinearity=silu;sgm.modules.diffusionmodules.openaimodel.th=sd_hijack_unet.th
	if current_optimizer is not _A:current_optimizer.undo();current_optimizer=_A
	B=option or shared.opts.cross_attention_optimization
	if B==C and len(optimizers)>0:A=next(iter([A for A in optimizers if A.cmd_opt and getattr(shared.cmd_opts,A.cmd_opt,_B)]),optimizers[0])
	else:A=next(iter([A for A in optimizers if A.title()==B]),_A)
	if B=='None':A=_A
	elif B==C and shared.cmd_opts.disable_opt_split_attention:A=_A
	elif A is _A:A=optimizers[0]
	if A is not _A:print(f"Applying attention optimization: {A.name}... ",end='');A.apply();print('done.');current_optimizer=A;return current_optimizer.name
	else:print('Disabling attention optimization');return''
def undo_optimizations():ldm.modules.diffusionmodules.model.nonlinearity=diffusionmodules_model_nonlinearity;ldm.modules.attention.CrossAttention.forward=hypernetwork.attention_CrossAttention_forward;ldm.modules.diffusionmodules.model.AttnBlock.forward=diffusionmodules_model_AttnBlock_forward;sgm.modules.diffusionmodules.model.nonlinearity=diffusionmodules_model_nonlinearity;sgm.modules.attention.CrossAttention.forward=hypernetwork.attention_CrossAttention_forward;sgm.modules.diffusionmodules.model.AttnBlock.forward=diffusionmodules_model_AttnBlock_forward
def fix_checkpoint():"checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want\n    checkpoints to be added when not training (there's a warning)"
def weighted_loss(sd_model,pred,target,mean=True):
	B=sd_model;A=B._old_get_loss(pred,target,mean=_B);C=getattr(B,'_custom_loss_weight',_A)
	if C is not _A:A*=C
	return A.mean()if mean else A
def weighted_forward(sd_model,x,c,w,*C,**D):
	B='_old_get_loss';A=sd_model
	try:
		A._custom_loss_weight=w
		if not hasattr(A,B):A._old_get_loss=A.get_loss
		A.get_loss=MethodType(weighted_loss,A);return A.forward(x,c,*C,**D)
	finally:
		try:del A._custom_loss_weight
		except AttributeError:pass
		if hasattr(A,B):A.get_loss=A._old_get_loss;del A._old_get_loss
def apply_weighted_forward(sd_model):A=sd_model;A.weighted_forward=MethodType(weighted_forward,A)
def undo_weighted_forward(sd_model):
	try:del sd_model.weighted_forward
	except AttributeError:pass
class StandardDemoModelHijack:
	fixes=_A;layers=_A;circular_enabled=_B;clip=_A;optimization_method=_A
	def __init__(A):import modules.textual_inversion.textual_inversion;A.extra_generation_params={};A.comments=[];A.embedding_db=modules.textual_inversion.textual_inversion.EmbeddingDatabase();A.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)
	def apply_optimizations(A,option=_A):
		try:A.optimization_method=apply_optimizations(option)
		except Exception as B:errors.display(B,'applying cross attention optimization');undo_optimizations()
	def hijack(A,m):
		B=getattr(m,_C,_A)
		if B:
			F=[]
			for E in range(len(B.embedders)):
				C=B.embedders[E];G=type(C).__name__
				if G=='FrozenOpenCLIPEmbedder':C.model.token_embedding=EmbeddingsWithFixes(C.model.token_embedding,A);B.embedders[E]=sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(C,A);F.append(B.embedders[E])
				if G=='FrozenCLIPEmbedder':D=C.transformer.text_model.embeddings;D.token_embedding=EmbeddingsWithFixes(D.token_embedding,A);B.embedders[E]=sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords(C,A);F.append(B.embedders[E])
				if G=='FrozenOpenCLIPEmbedder2':C.model.token_embedding=EmbeddingsWithFixes(C.model.token_embedding,A,textual_inversion_key='clip_g');B.embedders[E]=sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(C,A);F.append(B.embedders[E])
			if len(F)==1:m.cond_stage_model=F[0]
			else:m.cond_stage_model=B
		if type(m.cond_stage_model)==xlmr.BertSeriesModelWithTransformation:D=m.cond_stage_model.roberta.embeddings;D.token_embedding=EmbeddingsWithFixes(D.word_embeddings,A);m.cond_stage_model=sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model,A)
		elif type(m.cond_stage_model)==ldm.modules.encoders.modules.FrozenCLIPEmbedder:D=m.cond_stage_model.transformer.text_model.embeddings;D.token_embedding=EmbeddingsWithFixes(D.token_embedding,A);m.cond_stage_model=sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model,A)
		elif type(m.cond_stage_model)==ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:m.cond_stage_model.model.token_embedding=EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding,A);m.cond_stage_model=sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model,A)
		apply_weighted_forward(m)
		if m.cond_stage_key=='edit':sd_hijack_unet.hijack_ddpm_edit()
		A.apply_optimizations();A.clip=m.cond_stage_model
		def H(el):
			B=[H(A)for A in el.children()];A=[el]
			for C in B:A+=C
			return A
		A.layers=H(m)
		if not hasattr(ldm.modules.diffusionmodules.openaimodel,'copy_of_UNetModel_forward_for_webui'):ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui=ldm.modules.diffusionmodules.openaimodel.UNetModel.forward
		ldm.modules.diffusionmodules.openaimodel.UNetModel.forward=sd_unet.UNetModel_forward
	def undo_hijack(C,m):
		F='cond_stage_model';B=getattr(m,_C,_A)
		if B:
			for D in range(len(B.embedders)):
				A=B.embedders[D]
				if isinstance(A,(sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords,sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):A.wrapped.model.token_embedding=A.wrapped.model.token_embedding.wrapped;B.embedders[D]=A.wrapped
				if isinstance(A,sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):A.wrapped.transformer.text_model.embeddings.token_embedding=A.wrapped.transformer.text_model.embeddings.token_embedding.wrapped;B.embedders[D]=A.wrapped
			if hasattr(m,F):delattr(m,F)
		elif type(m.cond_stage_model)==sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:m.cond_stage_model=m.cond_stage_model.wrapped
		elif type(m.cond_stage_model)==sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
			m.cond_stage_model=m.cond_stage_model.wrapped;E=m.cond_stage_model.transformer.text_model.embeddings
			if type(E.token_embedding)==EmbeddingsWithFixes:E.token_embedding=E.token_embedding.wrapped
		elif type(m.cond_stage_model)==sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:m.cond_stage_model.wrapped.model.token_embedding=m.cond_stage_model.wrapped.model.token_embedding.wrapped;m.cond_stage_model=m.cond_stage_model.wrapped
		undo_optimizations();undo_weighted_forward(m);C.apply_circular(_B);C.layers=_A;C.clip=_A;ldm.modules.diffusionmodules.openaimodel.UNetModel.forward=ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui
	def apply_circular(A,enable):
		B=enable
		if A.circular_enabled==B:return
		A.circular_enabled=B
		for C in[A for A in A.layers if type(A)==torch.nn.Conv2d]:C.padding_mode=_D if B else'zeros'
	def clear_comments(A):A.comments=[];A.extra_generation_params={}
	def get_prompt_lengths(A,text):
		if A.clip is _A:return'-','-'
		C,B=A.clip.process_texts([text]);return B,A.clip.get_target_prompt_token_count(B)
	def redo_hijack(A,m):A.undo_hijack(m);A.hijack(m)
class EmbeddingsWithFixes(torch.nn.Module):
	def __init__(A,wrapped,embeddings,textual_inversion_key='clip_l'):super().__init__();A.wrapped=wrapped;A.embeddings=embeddings;A.textual_inversion_key=textual_inversion_key
	def forward(B,input_ids):
		C=B.embeddings.fixes;B.embeddings.fixes=_A;F=B.wrapped(input_ids)
		if C is _A or len(C)==0 or max([len(A)for A in C])==0:return F
		G=[]
		for(J,A)in zip(C,F):
			for(D,E)in J:K=E.vec[B.textual_inversion_key]if isinstance(E.vec,dict)else E.vec;H=devices.cond_cast_unet(K);I=min(A.shape[0]-D-1,H.shape[0]);A=torch.cat([A[0:D+1],H[0:I],A[D+1+I:]])
			G.append(A)
		return torch.stack(G)
def add_circular_option_to_conv_2d():
	A=torch.nn.Conv2d.__init__
	def B(self,*B,**C):return A(self,*B,padding_mode=_D,**C)
	torch.nn.Conv2d.__init__=B
model_hijack=StandardDemoModelHijack()
def register_buffer(self,name,attr):
	'\n    Fix register buffer bug for Mac OS.\n    ';A=attr
	if type(A)==torch.Tensor:
		if A.device!=devices.device:A=A.to(device=devices.device,dtype=torch.float32 if devices.device.type=='mps'else _A)
	setattr(self,name,A)
ldm.models.diffusion.ddim.DDIMSampler.register_buffer=register_buffer
ldm.models.diffusion.plms.PLMSSampler.register_buffer=register_buffer