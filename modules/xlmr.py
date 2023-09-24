_E='projection_state'
_D='absolute'
_C=False
_B=True
_A=None
from transformers import BertPreTrainedModel,BertConfig
import torch.nn as nn,torch
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers import XLMRobertaModel,XLMRobertaTokenizer
from typing import Optional
class BertSeriesConfig(BertConfig):
	def __init__(A,vocab_size=30522,hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072,hidden_act='gelu',hidden_dropout_prob=.1,attention_probs_dropout_prob=.1,max_position_embeddings=512,type_vocab_size=2,initializer_range=.02,layer_norm_eps=1e-12,pad_token_id=0,position_embedding_type=_D,use_cache=_B,classifier_dropout=_A,project_dim=512,pooler_fn='average',learn_encoder=_C,model_type='bert',**B):super().__init__(vocab_size,hidden_size,num_hidden_layers,num_attention_heads,intermediate_size,hidden_act,hidden_dropout_prob,attention_probs_dropout_prob,max_position_embeddings,type_vocab_size,initializer_range,layer_norm_eps,pad_token_id,position_embedding_type,use_cache,classifier_dropout,**B);A.project_dim=project_dim;A.pooler_fn=pooler_fn;A.learn_encoder=learn_encoder
class RobertaSeriesConfig(XLMRobertaConfig):
	def __init__(A,pad_token_id=1,bos_token_id=0,eos_token_id=2,project_dim=512,pooler_fn='cls',learn_encoder=_C,**B):super().__init__(pad_token_id=pad_token_id,bos_token_id=bos_token_id,eos_token_id=eos_token_id,**B);A.project_dim=project_dim;A.pooler_fn=pooler_fn;A.learn_encoder=learn_encoder
class BertSeriesModelWithTransformation(BertPreTrainedModel):
	_keys_to_ignore_on_load_unexpected=['pooler'];_keys_to_ignore_on_load_missing=['position_ids','predictions.decoder.bias'];config_class=BertSeriesConfig
	def __init__(B,config=_A,**C):
		A=config
		if A is _A:A=XLMRobertaConfig();A.attention_probs_dropout_prob=.1;A.bos_token_id=0;A.eos_token_id=2;A.hidden_act='gelu';A.hidden_dropout_prob=.1;A.hidden_size=1024;A.initializer_range=.02;A.intermediate_size=4096;A.layer_norm_eps=1e-05;A.max_position_embeddings=514;A.num_attention_heads=16;A.num_hidden_layers=24;A.output_past=_B;A.pad_token_id=1;A.position_embedding_type=_D;A.type_vocab_size=1;A.use_cache=_B;A.vocab_size=250002;A.project_dim=768;A.learn_encoder=_C
		super().__init__(A);B.roberta=XLMRobertaModel(A);B.transformation=nn.Linear(A.hidden_size,A.project_dim);B.pre_LN=nn.LayerNorm(A.hidden_size,eps=A.layer_norm_eps);B.tokenizer=XLMRobertaTokenizer.from_pretrained('xlm-roberta-large');B.pooler=lambda x:x[:,0];B.post_init()
	def encode(B,c):C='attention_mask';D='input_ids';E=next(B.parameters()).device;A=B.tokenizer(c,truncation=_B,max_length=77,return_length=_C,return_overflowing_tokens=_C,padding='max_length',return_tensors='pt');A[D]=torch.tensor(A[D]).to(E);A[C]=torch.tensor(A[C]).to(E);F=B(**A);return F[_E]
	def forward(A,input_ids=_A,attention_mask=_A,token_type_ids=_A,position_ids=_A,head_mask=_A,inputs_embeds=_A,encoder_hidden_states=_A,encoder_attention_mask=_A,output_attentions=_A,return_dict=_A,output_hidden_states=_A):'\n        ';C=return_dict;C=C if C is not _A else A.config.use_return_dict;B=A.roberta(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,position_ids=position_ids,head_mask=head_mask,inputs_embeds=inputs_embeds,encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=encoder_attention_mask,output_attentions=output_attentions,output_hidden_states=_B,return_dict=C);E=B[0];F=A.pre_LN(E);D=A.pooler(F);D=A.transformation(D);G=A.transformation(B.last_hidden_state);return{'pooler_output':D,'last_hidden_state':B.last_hidden_state,'hidden_states':B.hidden_states,'attentions':B.attentions,_E:G,'sequence_out':E}
class RobertaSeriesModelWithTransformation(BertSeriesModelWithTransformation):base_model_prefix='roberta';config_class=RobertaSeriesConfig