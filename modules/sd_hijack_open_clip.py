_C='Old emphasis implementation not supported for Open Clip'
_B='<end_of_text>'
_A='<start_of_text>'
import open_clip.tokenizer,torch
from modules import sd_hijack_clip,devices
from modules.shared import opts
tokenizer=open_clip.tokenizer._tokenizer
class FrozenOpenCLIPEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
	def __init__(A,wrapped,hijack):super().__init__(wrapped,hijack);A.comma_token=[B for(A,B)in tokenizer.encoder.items()if A==',</w>'][0];A.id_start=tokenizer.encoder[_A];A.id_end=tokenizer.encoder[_B];A.id_pad=0
	def tokenize(B,texts):assert not opts.use_old_emphasis_implementation,_C;A=[tokenizer.encode(A)for A in texts];return A
	def encode_with_transformers(A,tokens):B=A.wrapped.encode_with_transformer(tokens);return B
	def encode_embedding_init_text(B,init_text,nvpt):A=tokenizer.encode(init_text);A=torch.asarray([A],device=devices.device,dtype=torch.int);C=B.wrapped.model.token_embedding.wrapped(A).squeeze(0);return C
class FrozenOpenCLIPEmbedder2WithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
	def __init__(A,wrapped,hijack):super().__init__(wrapped,hijack);A.comma_token=[B for(A,B)in tokenizer.encoder.items()if A==',</w>'][0];A.id_start=tokenizer.encoder[_A];A.id_end=tokenizer.encoder[_B];A.id_pad=0
	def tokenize(B,texts):assert not opts.use_old_emphasis_implementation,_C;A=[tokenizer.encode(A)for A in texts];return A
	def encode_with_transformers(A,tokens):
		B=A.wrapped.encode_with_transformer(tokens);C=B[A.wrapped.layer];D=B.get('pooled')
		if D is not None:C.pooled=D
		return C
	def encode_embedding_init_text(B,init_text,nvpt):A=tokenizer.encode(init_text);A=torch.asarray([A],device=devices.device,dtype=torch.int);C=B.wrapped.model.token_embedding.wrapped(A.to(B.wrapped.model.token_embedding.wrapped.weight.device)).squeeze(0);return C