_D='utf-8-sig'
_C=False
_B='{prompt}'
_A=True
import csv,os,os.path,re,typing,shutil
class PromptStyle(typing.NamedTuple):name:str;prompt:str;negative_prompt:str
def merge_prompts(style_prompt,prompt):
	B=prompt;A=style_prompt
	if _B in A:C=A.replace(_B,B)
	else:D=filter(None,(B.strip(),A.strip()));C=', '.join(D)
	return C
def apply_styles_to_prompt(prompt,styles):
	A=prompt
	for B in styles:A=merge_prompts(B,A)
	return A
re_spaces=re.compile('  +')
def extract_style_text_from_prompt(style_text,prompt):
	A=prompt;B=re.sub(re_spaces,' ',A.strip());C=re.sub(re_spaces,' ',style_text.strip())
	if _B in C:
		D,E=C.split(_B,2)
		if B.startswith(D)and B.endswith(E):A=B[len(D):len(B)-len(E)];return _A,A
	elif B.endswith(C):
		A=B[:len(B)-len(C)]
		if A.endswith(', '):A=A[:-2]
		return _A,A
	return _C,A
def extract_style_from_prompts(style,prompt,negative_prompt):
	A=negative_prompt;B=prompt;C=style
	if not C.prompt and not C.negative_prompt:return _C,B,A
	D,E=extract_style_text_from_prompt(C.prompt,B)
	if not D:return _C,B,A
	F,G=extract_style_text_from_prompt(C.negative_prompt,A)
	if not F:return _C,B,A
	return _A,E,G
class StyleDatabase:
	def __init__(A,path):A.no_style=PromptStyle('None','','');A.styles={};A.path=path;A.reload()
	def reload(B):
		C='name';D='prompt';B.styles.clear()
		if not os.path.exists(B.path):return
		with open(B.path,'r',encoding=_D,newline='')as E:
			F=csv.DictReader(E,skipinitialspace=_A)
			for A in F:G=A[D]if D in A else A['text'];H=A.get('negative_prompt','');B.styles[A[C]]=PromptStyle(A[C],G,H)
	def get_style_prompts(A,styles):return[A.styles.get(B,A.no_style).prompt for B in styles]
	def get_negative_style_prompts(A,styles):return[A.styles.get(B,A.no_style).negative_prompt for B in styles]
	def apply_styles_to_prompt(A,prompt,styles):return apply_styles_to_prompt(prompt,[A.styles.get(B,A.no_style).prompt for B in styles])
	def apply_negative_styles_to_prompt(A,prompt,styles):return apply_styles_to_prompt(prompt,[A.styles.get(B,A.no_style).negative_prompt for B in styles])
	def save_styles(C,path):
		A=path
		if os.path.exists(A):shutil.copy(A,f"{A}.bak")
		with open(A,'w',encoding=_D,newline='')as D:B=csv.DictWriter(D,fieldnames=PromptStyle._fields);B.writeheader();B.writerows(A._asdict()for(B,A)in C.styles.items())
	def extract_styles_from_prompt(G,prompt,negative_prompt):
		B=negative_prompt;C=prompt;D=[];E=list(G.styles.values())
		while _A:
			A=None
			for F in E:
				H,I,J=extract_style_from_prompts(F,C,B)
				if H:A=F;C=I;B=J;break
			if not A:break
			E.remove(A);D.append(A.name)
		return list(reversed(D)),C,B