from __future__ import annotations
_D='crossattn'
_C=False
_B=1.
_A=None
import re
from collections import namedtuple
from typing import List
import lark
schedule_parser=lark.Lark('\n!start: (prompt | /[][():]/+)*\nprompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*\n!emphasized: "(" prompt ")"\n        | "(" prompt ":" prompt ")"\n        | "[" prompt "]"\nscheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"\nalternate: "[" prompt ("|" [prompt])+ "]"\nWHITESPACE: /\\s+/\nplain: /([^\\\\\\[\\]():|]|\\\\.)+/\n%import common.SIGNED_NUMBER -> NUMBER\n')
def get_learned_conditioning_prompt_schedules(prompts,base_steps,hires_steps=_A,use_old_scheduling=_C):
	'\n    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]\n    >>> g("test")\n    [[10, \'test\']]\n    >>> g("a [b:3]")\n    [[3, \'a \'], [10, \'a b\']]\n    >>> g("a [b: 3]")\n    [[3, \'a \'], [10, \'a b\']]\n    >>> g("a [[[b]]:2]")\n    [[2, \'a \'], [10, \'a [[b]]\']]\n    >>> g("[(a:2):3]")\n    [[3, \'\'], [10, \'(a:2)\']]\n    >>> g("a [b : c : 1] d")\n    [[1, \'a b  d\'], [10, \'a  c  d\']]\n    >>> g("a[b:[c:d:2]:1]e")\n    [[1, \'abe\'], [2, \'ace\'], [10, \'ade\']]\n    >>> g("a [unbalanced")\n    [[10, \'a [unbalanced\']]\n    >>> g("a [b:.5] c")\n    [[5, \'a  c\'], [10, \'a b c\']]\n    >>> g("a [{b|d{:.5] c")  # not handling this right now\n    [[5, \'a  c\'], [10, \'a {b|d{ c\']]\n    >>> g("((a][:b:c [d:3]")\n    [[3, \'((a][:b:c \'], [10, \'((a][:b:c d\']]\n    >>> g("[a|(b:1.1)]")\n    [[1, \'a\'], [2, \'(b:1.1)\'], [3, \'a\'], [4, \'(b:1.1)\'], [5, \'a\'], [6, \'(b:1.1)\'], [7, \'a\'], [8, \'(b:1.1)\'], [9, \'a\'], [10, \'(b:1.1)\']]\n    >>> g("[fe|]male")\n    [[1, \'female\'], [2, \'male\'], [3, \'female\'], [4, \'male\'], [5, \'female\'], [6, \'male\'], [7, \'female\'], [8, \'male\'], [9, \'female\'], [10, \'male\']]\n    >>> g("[fe|||]male")\n    [[1, \'female\'], [2, \'male\'], [3, \'male\'], [4, \'male\'], [5, \'female\'], [6, \'male\'], [7, \'male\'], [8, \'male\'], [9, \'female\'], [10, \'male\']]\n    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]\n    >>> g("a [b:.5] c")\n    [[10, \'a b c\']]\n    >>> g("a [b:1.5] c")\n    [[5, \'a  c\'], [10, \'a b c\']]\n    ';E=use_old_scheduling;D=hires_steps;C=base_steps;B=prompts
	if D is _A or E:F=0;G=0;A=C
	else:F=C;G=_B;A=D
	def H(steps,tree):
		B=steps;D=[B]
		class A(lark.Visitor):
			def scheduled(I,tree):
				C=tree;H=C.children[-2];A=float(H)
				if E:A=A*B if A<1 else A
				elif'.'in H:A=(A-G)*B
				else:A=A-F
				C.children[-2]=min(B,int(A))
				if C.children[-2]>=1:D.append(C.children[-2])
			def alternate(A,tree):D.extend(range(1,B+1))
		A().visit(tree);return sorted(set(D))
	def I(step,tree):
		class A(lark.Transformer):
			def scheduled(E,args):A,B,C,D,C=args;yield A or()if step<=D else B
			def alternate(B,args):A=args;A=[''if not A else A for A in A];yield A[(step-1)%len(A)]
			def start(B,args):
				def A(x):
					if isinstance(x,str):yield x
					else:
						for B in x:yield from A(B)
				return''.join(A(args))
			def plain(A,args):yield args[0].value
			def __default__(B,data,children,meta):
				for A in children:yield A
		return A().transform(tree)
	def J(prompt):
		B=prompt
		try:C=schedule_parser.parse(B)
		except lark.exceptions.LarkError:
			if 0:import traceback as D;D.print_exc()
			return[[A,B]]
		return[[A,I(A,C)]for A in H(A,C)]
	K={A:J(A)for A in set(B)};return[K[A]for A in B]
ScheduledPromptConditioning=namedtuple('ScheduledPromptConditioning',['end_at_step','cond'])
class SdConditioning(list):
	"\n    A list with prompts for stable diffusion's conditioner model.\n    Can also specify width and height of created image - SDXL needs it.\n    "
	def __init__(B,prompts,is_negative_prompt=_C,width=_A,height=_A,copy_from=_A):
		C=prompts;A=copy_from;super().__init__();B.extend(C)
		if A is _A:A=C
		B.is_negative_prompt=is_negative_prompt or getattr(A,'is_negative_prompt',_C);B.width=width or getattr(A,'width',_A);B.height=height or getattr(A,'height',_A)
def get_learned_conditioning(model,prompts,steps,hires_steps=_A,use_old_scheduling=_C):
	"converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),\n    and the sampling step at which this condition is to be replaced by the next one.\n\n    Input:\n    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)\n\n    Output:\n    [\n        [\n            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))\n        ],\n        [\n            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),\n            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))\n        ]\n    ]\n    ";A=prompts;B=[];K=get_learned_conditioning_prompt_schedules(A,steps,hires_steps,use_old_scheduling);E={}
	for(F,G)in zip(A,K):
		H=E.get(F,_A)
		if H is not _A:B.append(H);continue
		L=SdConditioning([A[1]for A in G],copy_from=A);C=model.get_learned_conditioning(L);D=[]
		for(I,(M,N))in enumerate(G):
			if isinstance(C,dict):J={A:B[I]for(A,B)in C.items()}
			else:J=C[I]
			D.append(ScheduledPromptConditioning(M,J))
		E[F]=D;B.append(D)
	return B
re_AND=re.compile('\\bAND\\b')
re_weight=re.compile('^((?:\\s|.)*?)(?:\\s*:\\s*([-+]?(?:\\d+\\.?|\\d*\\.\\d+)))?\\s*$')
def get_multicond_prompt_list(prompts):
	F=prompts;G=[];D={};A=SdConditioning(F);A.clear()
	for K in F:
		L=re_AND.split(K);H=[]
		for I in L:
			J=re_weight.search(I);E,B=J.groups()if J is not _A else(I,_B);B=float(B)if B is not _A else _B;C=D.get(E,_A)
			if C is _A:C=len(A);A.append(E);D[E]=C
			H.append((C,B))
		G.append(H)
	return G,A,D
class ComposableScheduledPromptConditioning:
	def __init__(A,schedules,weight=_B):A.schedules=schedules;A.weight=weight
class MulticondLearnedConditioning:
	def __init__(A,shape,batch):A.shape=shape;A.batch=batch
def get_multicond_learned_conditioning(model,prompts,steps,hires_steps=_A,use_old_scheduling=_C):
	'same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.\n    For each prompt, the list is obtained by splitting the prompt using the AND separator.\n\n    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/\n    ';A=prompts;C,D,G=get_multicond_prompt_list(A);E=get_learned_conditioning(model,D,steps,hires_steps,use_old_scheduling);B=[]
	for F in C:B.append([ComposableScheduledPromptConditioning(E[A],B)for(A,B)in F])
	return MulticondLearnedConditioning(shape=(len(A),),batch=B)
class DictWithShape(dict):
	def __init__(A,x,shape):super().__init__();A.update(x)
	@property
	def shape(self):return self[_D].shape
def reconstruct_cond_batch(c,current_step):
	A=c[0][0].cond;E=isinstance(A,dict)
	if E:F=A;B={B:torch.zeros((len(c),)+A.shape,device=A.device,dtype=A.dtype)for(B,A)in F.items()};B=DictWithShape(B,(len(c),)+F[_D].shape)
	else:B=torch.zeros((len(c),)+A.shape,device=A.device,dtype=A.dtype)
	for(G,C)in enumerate(c):
		D=0
		for(H,I)in enumerate(C):
			if current_step<=I.end_at_step:D=H;break
		if E:
			for(J,A)in C[D].cond.items():B[J][G]=A
		else:B[G]=C[D].cond
	return B
def stack_conds(tensors):
	A=tensors;C=max([A.shape[0]for A in A])
	for B in range(len(A)):
		if A[B].shape[0]!=C:D=A[B][-1:];E=D.repeat([C-A[B].shape[0],1]);A[B]=torch.vstack([A[B],E])
	return torch.stack(A)
def reconstruct_multicond_batch(c,current_step):
	D=c.batch[0][0].schedules[0].cond;A=[];E=[]
	for H in c.batch:
		F=[]
		for C in H:
			G=0
			for(I,J)in enumerate(C.schedules):
				if current_step<=J.end_at_step:G=I;break
			F.append((len(A),C.weight));A.append(C.schedules[G].cond)
		E.append(F)
	if isinstance(A[0],dict):K=list(A[0].keys());B={B:stack_conds([A[B]for A in A])for B in K};B=DictWithShape(B,B[_D].shape)
	else:B=stack_conds(A).to(device=D.device,dtype=D.dtype)
	return E,B
re_attention=re.compile('\n\\\\\\(|\n\\\\\\)|\n\\\\\\[|\n\\\\]|\n\\\\\\\\|\n\\\\|\n\\(|\n\\[|\n:\\s*([+-]?[.\\d]+)\\s*\\)|\n\\)|\n]|\n[^\\\\()\\[\\]:]+|\n:\n',re.X)
re_break=re.compile('\\s*\\bBREAK\\b\\s*',re.S)
def parse_prompt_attention(text):
	"\n    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.\n    Accepted tokens are:\n      (abc) - increases attention to abc by a multiplier of 1.1\n      (abc:3.12) - increases attention to abc by a multiplier of 3.12\n      [abc] - decreases attention to abc by a multiplier of 1.1\n      \\( - literal character '('\n      \\[ - literal character '['\n      \\) - literal character ')'\n      \\] - literal character ']'\n      \\ - literal character ''\n      anything else - just text\n\n    >>> parse_prompt_attention('normal text')\n    [['normal text', 1.0]]\n    >>> parse_prompt_attention('an (important) word')\n    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]\n    >>> parse_prompt_attention('(unbalanced')\n    [['unbalanced', 1.1]]\n    >>> parse_prompt_attention('\\(literal\\]')\n    [['(literal]', 1.0]]\n    >>> parse_prompt_attention('(unnecessary)(parens)')\n    [['unnecessaryparens', 1.1]]\n    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')\n    [['a ', 1.0],\n     ['house', 1.5730000000000004],\n     [' ', 1.1],\n     ['on', 1.0],\n     [' a ', 1.1],\n     ['hill', 0.55],\n     [', sun, ', 1.1],\n     ['sky', 1.4641000000000006],\n     ['.', 1.1]]\n    ";B=text;A=[];D=[];F=[];H=1.1;I=1/1.1
	def E(start_position,multiplier):
		for B in range(start_position,len(A)):A[B][1]*=multiplier
	for J in re_attention.finditer(B):
		B=J.group(0);K=J.group(1)
		if B.startswith('\\'):A.append([B[1:],_B])
		elif B=='(':D.append(len(A))
		elif B=='[':F.append(len(A))
		elif K is not _A and D:E(D.pop(),float(K))
		elif B==')'and D:E(D.pop(),H)
		elif B==']'and F:E(F.pop(),I)
		else:
			L=re.split(re_break,B)
			for(C,M)in enumerate(L):
				if C>0:A.append(['BREAK',-1])
				A.append([M,_B])
	for G in D:E(G,H)
	for G in F:E(G,I)
	if len(A)==0:A=[['',_B]]
	C=0
	while C+1<len(A):
		if A[C][1]==A[C+1][1]:A[C][0]+=A[C+1][0];A.pop(C+1)
		else:C+=1
	return A
if __name__=='__main__':import doctest;doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
else:import torch