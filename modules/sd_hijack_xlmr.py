import torch
from modules import sd_hijack_clip,devices
class FrozenXLMREmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):
	def __init__(A,wrapped,hijack):B=wrapped;super().__init__(B,hijack);A.id_start=B.config.bos_token_id;A.id_end=B.config.eos_token_id;A.id_pad=B.config.pad_token_id;A.comma_token=A.tokenizer.get_vocab().get(',',None)
	def encode_with_transformers(B,tokens):A=tokens;C=(A!=B.id_pad).to(device=A.device,dtype=torch.int64);D=B.wrapped(input_ids=A,attention_mask=C);E=D['projection_state'];return E
	def encode_embedding_init_text(A,init_text,nvpt):B=A.wrapped.roberta.embeddings;C=A.wrapped.tokenizer(init_text,max_length=nvpt,return_tensors='pt',add_special_tokens=False)['input_ids'];D=B.token_embedding.wrapped(C.to(devices.device)).squeeze(0);return D