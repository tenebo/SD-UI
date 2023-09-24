_D='input_ids'
_C=None
_B=False
_A=1.
import math
from collections import namedtuple
import torch
from modules import prompt_parser,devices,sd_hijack
from modules.shared import opts
class PromptChunk:
	'\n    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.\n    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.\n    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,\n    so just 75 tokens from prompt.\n    '
	def __init__(A):A.tokens=[];A.multipliers=[];A.fixes=[]
PromptChunkFix=namedtuple('PromptChunkFix',['offset','embedding'])
"An object of this type is a marker showing that textual inversion embedding's vectors have to placed at offset in the prompt\nchunk. Thos objects are found in PromptChunk.fixes and, are placed into FrozenCLIPEmbedderWithCustomWordsBase.hijack.fixes, and finally\nare applied by sd_hijack.EmbeddingsWithFixes's forward function."
class FrozenCLIPEmbedderWithCustomWordsBase(torch.nn.Module):
	'A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to\n    have unlimited prompt length and assign weights to tokens in prompt.\n    '
	def __init__(A,wrapped,hijack):B=wrapped;super().__init__();A.wrapped=B;'Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,\n        depending on model.';A.hijack=hijack;A.chunk_length=75;A.is_trainable=getattr(B,'is_trainable',_B);A.input_key=getattr(B,'input_key','txt');A.legacy_ucg_val=_C
	def empty_chunk(A):'creates an empty PromptChunk and returns it';B=PromptChunk();B.tokens=[A.id_start]+[A.id_end]*(A.chunk_length+1);B.multipliers=[_A]*(A.chunk_length+2);return B
	def get_target_prompt_token_count(A,token_count):'returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented';return math.ceil(max(token_count,1)/A.chunk_length)*A.chunk_length
	def tokenize(A,texts):'Converts a batch of texts into a batch of token ids';raise NotImplementedError
	def encode_with_transformers(A,tokens):'\n        converts a batch of token ids (in python lists) into a single tensor with numeric respresentation of those tokens;\n        All python lists with tokens are assumed to have same length, usually 77.\n        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on\n        model - can be 768 and 1024.\n        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).\n        ';raise NotImplementedError
	def encode_embedding_init_text(A,init_text,nvpt):"Converts text into a tensor with this text's tokens' embeddings. Note that those are embeddings before they are passed through\n        transformers. nvpt is used as a maximum length in tokens. If text produces less teokens than nvpt, only this many is returned.";raise NotImplementedError
	def tokenize_line(B,line):
		'\n        this transforms a single prompt into a list of PromptChunk objects - as many as needed to\n        represent the prompt.\n        Returns the list and the total number of tokens in the prompt.\n        '
		if opts.enable_emphasis:H=prompt_parser.parse_prompt_attention(line)
		else:H=[[line,_A]]
		O=B.tokenize([A for(A,B)in H]);I=[];A=PromptChunk();F=0;C=-1
		def D(is_last=_B):
			"puts current chunk into the list of results and produces the next one - empty;\n            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count";nonlocal F;nonlocal C;nonlocal A
			if is_last:F+=len(A.tokens)
			else:F+=B.chunk_length
			D=B.chunk_length-len(A.tokens)
			if D>0:A.tokens+=[B.id_end]*D;A.multipliers+=[_A]*D
			A.tokens=[B.id_start]+A.tokens+[B.id_end];A.multipliers=[_A]+A.multipliers+[_A];C=-1;I.append(A);A=PromptChunk()
		for(J,(P,K))in zip(O,H):
			if P=='BREAK'and K==-1:D();continue
			E=0
			while E<len(J):
				N=J[E]
				if N==B.comma_token:C=len(A.tokens)
				elif opts.comma_padding_backtrack!=0 and len(A.tokens)==B.chunk_length and C!=-1 and len(A.tokens)-C<=opts.comma_padding_backtrack:G=C+1;Q=A.tokens[G:];R=A.multipliers[G:];A.tokens=A.tokens[:G];A.multipliers=A.multipliers[:G];D();A.tokens=Q;A.multipliers=R
				if len(A.tokens)==B.chunk_length:D()
				L,S=B.hijack.embedding_db.find_embedding_at_position(J,E)
				if L is _C:A.tokens.append(N);A.multipliers.append(K);E+=1;continue
				M=int(L.vectors)
				if len(A.tokens)+M>B.chunk_length:D()
				A.fixes.append(PromptChunkFix(len(A.tokens),L));A.tokens+=[0]*M;A.multipliers+=[K]*M;E+=S
		if A.tokens or not I:D(is_last=True)
		return I,F
	def process_texts(F,texts):
		'\n        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum\n        length, in tokens, of all texts.\n        ';B=0;C={};E=[]
		for A in texts:
			if A in C:D=C[A]
			else:D,G=F.tokenize_line(A);B=max(G,B);C[A]=D
			E.append(D)
		return E,B
	def forward(A,texts):
		'\n        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.\n        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will\n        be a multiple of 77; and C is dimensionality of each token - for SD1 it\'s 768, for SD2 it\'s 1024, and for SDXL it\'s 1280.\n        An example shape returned by this function can be: (2, 77, 768).\n        For SDXL, instead of returning one tensor avobe, it returns a tuple with two: the other one with shape (B, 1280) with pooled values.\n        Ourui usually sends just one text at a time through this function - the only time when texts is an array with more than one elemenet\n        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"\n        ';I=texts;E='TI hashes'
		if opts.use_old_emphasis_implementation:import modules.sd_hijack_clip_old;return modules.sd_hijack_clip_old.forward_old(A,I)
		J,R=A.process_texts(I);F={};M=max([len(A)for A in J]);B=[]
		for K in range(M):
			G=[B[K]if K<len(B)else A.empty_chunk()for B in J];N=[A.tokens for A in G];O=[A.multipliers for A in G];A.hijack.fixes=[A.fixes for A in G]
			for P in A.hijack.fixes:
				for(S,C)in P:F[C.name]=C
			Q=A.process_tokens(N,O);B.append(Q)
		if opts.textual_inversion_add_hashes_to_infotext and F:
			D=[]
			for(H,C)in F.items():
				L=C.shorthash
				if not L:continue
				H=H.replace(':','').replace(',','');D.append(f"{H}: {L}")
			if D:
				if A.hijack.extra_generation_params.get(E):D.append(A.hijack.extra_generation_params.get(E))
				A.hijack.extra_generation_params[E]=', '.join(D)
		if getattr(A.wrapped,'return_pooled',_B):return torch.hstack(B),B[0].pooled
		else:return torch.hstack(B)
	def process_tokens(B,remade_batch_tokens,batch_multipliers):
		'\n        sends one single prompt chunk to be encoded by transformers neural network.\n        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually\n        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.\n        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier\n        corresponds to one token.\n        ';D=remade_batch_tokens;C=batch_multipliers;E=torch.asarray(D).to(devices.device)
		if B.id_end!=B.id_pad:
			for F in range(len(D)):H=D[F].index(B.id_end);E[F,H+1:E.shape[1]]=B.id_pad
		A=B.encode_with_transformers(E);G=getattr(A,'pooled',_C);C=torch.asarray(C).to(devices.device);I=A.mean();A=A*C.reshape(C.shape+(1,)).expand(A.shape);J=A.mean();A=A*(I/J)
		if G is not _C:A.pooled=G
		return A
class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
	def __init__(A,wrapped,hijack):
		D=wrapped;super().__init__(D,hijack);A.tokenizer=D.tokenizer;E=A.tokenizer.get_vocab();A.comma_token=E.get(',</w>',_C);A.token_mults={};F=[(A,B)for(A,B)in E.items()if'('in A or')'in A or'['in A or']'in A]
		for(G,H)in F:
			B=_A
			for C in G:
				if C=='[':B/=1.1
				if C==']':B*=1.1
				if C=='(':B*=1.1
				if C==')':B/=1.1
			if B!=_A:A.token_mults[H]=B
		A.id_start=A.wrapped.tokenizer.bos_token_id;A.id_end=A.wrapped.tokenizer.eos_token_id;A.id_pad=A.id_end
	def tokenize(A,texts):B=A.wrapped.tokenizer(texts,truncation=_B,add_special_tokens=_B)[_D];return B
	def encode_with_transformers(B,tokens):
		C=B.wrapped.transformer(input_ids=tokens,output_hidden_states=-opts.CLIP_stop_at_last_layers)
		if opts.CLIP_stop_at_last_layers>1:A=C.hidden_states[-opts.CLIP_stop_at_last_layers];A=B.wrapped.transformer.text_model.final_layer_norm(A)
		else:A=C.last_hidden_state
		return A
	def encode_embedding_init_text(A,init_text,nvpt):B=A.wrapped.transformer.text_model.embeddings;C=A.wrapped.tokenizer(init_text,max_length=nvpt,return_tensors='pt',add_special_tokens=_B)[_D];D=B.token_embedding.wrapped(C.to(B.token_embedding.wrapped.weight.device)).squeeze(0);return D
class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
	def __init__(A,wrapped,hijack):super().__init__(wrapped,hijack)
	def encode_with_transformers(A,tokens):
		B=A.wrapped.transformer(input_ids=tokens,output_hidden_states=A.wrapped.layer=='hidden')
		if A.wrapped.layer=='last':C=B.last_hidden_state
		else:C=B.hidden_states[A.wrapped.layer_idx]
		return C