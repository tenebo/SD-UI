from __future__ import annotations
_A='txt'
import torch,sgm.models.diffusion,sgm.modules.diffusionmodules.denoiser_scaling,sgm.modules.diffusionmodules.discretizer
from modules import devices,shared,prompt_parser
def get_learned_conditioning(self,batch):
	A=batch
	for F in self.conditioner.embedders:F.ucg_rate=.0
	C=getattr(A,'width',1024);D=getattr(A,'height',1024);E=getattr(A,'is_negative_prompt',False);G=shared.opts.sdxl_refiner_low_aesthetic_score if E else shared.opts.sdxl_refiner_high_aesthetic_score;B=dict(device=devices.device,dtype=devices.dtype);H={_A:A,'original_size_as_tuple':torch.tensor([D,C],**B).repeat(len(A),1),'crop_coords_top_left':torch.tensor([shared.opts.sdxl_crop_top,shared.opts.sdxl_crop_left],**B).repeat(len(A),1),'target_size_as_tuple':torch.tensor([D,C],**B).repeat(len(A),1),'aesthetic_score':torch.tensor([G],**B).repeat(len(A),1)};I=E and all(A==''for A in A);J=self.conditioner(H,force_zero_embeddings=[_A]if I else[]);return J
def apply_model(self,x,t,cond):return self.model(x,t,cond)
def get_first_stage_encoding(self,x):return x
sgm.models.diffusion.DiffusionEngine.get_learned_conditioning=get_learned_conditioning
sgm.models.diffusion.DiffusionEngine.apply_model=apply_model
sgm.models.diffusion.DiffusionEngine.get_first_stage_encoding=get_first_stage_encoding
def encode_embedding_init_text(self,init_text,nvpt):
	A=[]
	for B in[A for A in self.embedders if hasattr(A,'encode_embedding_init_text')]:C=B.encode_embedding_init_text(init_text,nvpt);A.append(C)
	return torch.cat(A,dim=1)
def tokenize(self,texts):
	for A in[A for A in self.embedders if hasattr(A,'tokenize')]:return A.tokenize(texts)
	raise AssertionError('no tokenizer available')
def process_texts(self,texts):
	for A in[A for A in self.embedders if hasattr(A,'process_texts')]:return A.process_texts(texts)
def get_target_prompt_token_count(self,token_count):
	for A in[A for A in self.embedders if hasattr(A,'get_target_prompt_token_count')]:return A.get_target_prompt_token_count(token_count)
sgm.modules.GeneralConditioner.encode_embedding_init_text=encode_embedding_init_text
sgm.modules.GeneralConditioner.tokenize=tokenize
sgm.modules.GeneralConditioner.process_texts=process_texts
sgm.modules.GeneralConditioner.get_target_prompt_token_count=get_target_prompt_token_count
def extend_sdxl(model):'this adds a bunch of parameters to make SDXL model look a bit more like SD1.5 to the rest of the codebase.';A=model;B=next(A.model.diffusion_model.parameters()).dtype;A.model.diffusion_model.dtype=B;A.model.conditioning_key='crossattn';A.cond_stage_key=_A;A.parameterization='v'if isinstance(A.denoiser.scaling,sgm.modules.diffusionmodules.denoiser_scaling.VScaling)else'eps';C=sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization();A.alphas_cumprod=torch.asarray(C.alphas_cumprod,device=devices.device,dtype=B);A.conditioner.wrapped=torch.nn.Module()
sgm.modules.attention.print=shared.ldm_print
sgm.modules.diffusionmodules.model.print=shared.ldm_print
sgm.modules.diffusionmodules.openaimodel.print=shared.ldm_print
sgm.modules.encoders.modules.print=shared.ldm_print
sgm.modules.attention.SDP_IS_AVAILABLE=True
sgm.modules.attention.XFORMERS_IS_AVAILABLE=False