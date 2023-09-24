_C=True
_B=False
_A=None
import json,sys,gradio as gr
from modules import errors
from modules.shared_cmd_options import cmd_opts
class OptionInfo:
	def __init__(A,default=_A,label='',component=_A,component_args=_A,onchange=_A,section=_A,refresh=_A,comment_before='',comment_after='',infotext=_A,restrict_api=_B):A.default=default;A.label=label;A.component=component;A.component_args=component_args;A.onchange=onchange;A.section=section;A.refresh=refresh;A.do_not_save=_B;A.comment_before=comment_before;'HTML text that will be added after label in UI';A.comment_after=comment_after;'HTML text that will be added before label in UI';A.infotext=infotext;A.restrict_api=restrict_api;'If True, the setting will not be accessible via API'
	def link(A,label,url):A.comment_before+=f"[<a href='{url}' target='_blank'>{label}</a>]";return A
	def js(A,label,js_func):A.comment_before+=f"[<a onclick='{js_func}(); return false'>{label}</a>]";return A
	def info(A,info):A.comment_after+=f"<span class='info'>({info})</span>";return A
	def html(A,html):A.comment_after+=html;return A
	def needs_restart(A):A.comment_after+=" <span class='info'>(requires restart)</span>";return A
	def needs_reload_ui(A):A.comment_after+=" <span class='info'>(requires Reload UI)</span>";return A
class OptionHTML(OptionInfo):
	def __init__(A,text):super().__init__(str(text).strip(),label='',component=lambda**A:gr.HTML(elem_classes='settings-info',**A));A.do_not_save=_C
def options_section(section_identifier,options_dict):
	A=options_dict
	for B in A.values():B.section=section_identifier
	return A
options_builtin_fields={'data_labels','data','restricted_opts','typemap'}
class Options:
	typemap={int:float}
	def __init__(A,data_labels,restricted_opts):A.data_labels=data_labels;A.data={A:B.default for(A,B)in A.data_labels.items()};A.restricted_opts=restricted_opts
	def __setattr__(B,key,value):
		C=value;A=key
		if A in options_builtin_fields:return super(Options,B).__setattr__(A,C)
		if B.data is not _A:
			if A in B.data or A in B.data_labels:
				assert not cmd_opts.freeze_settings,'changing settings is disabled';D=B.data_labels.get(A,_A)
				if D.do_not_save:return
				E=D.component_args if D else _A
				if isinstance(E,dict)and E.get('visible',_C)is _B:raise RuntimeError(f"not possible to set {A} because it is restricted")
				if cmd_opts.hide_ui_dir_config and A in B.restricted_opts:raise RuntimeError(f"not possible to set {A} because it is restricted")
				B.data[A]=C;return
		return super(Options,B).__setattr__(A,C)
	def __getattr__(A,item):
		B=item
		if B in options_builtin_fields:return super(Options,A).__getattribute__(B)
		if A.data is not _A:
			if B in A.data:return A.data[B]
		if B in A.data_labels:return A.data_labels[B].default
		return super(Options,A).__getattribute__(B)
	def set(B,key,value,is_api=_B,run_callbacks=_C):
		'sets an option and calls its onchange callback, returning True if the option changed and False otherwise';D=value;A=key;E=B.data.get(A,_A)
		if E==D:return _B
		C=B.data_labels[A]
		if C.do_not_save:return _B
		if is_api and C.restrict_api:return _B
		try:setattr(B,A,D)
		except RuntimeError:return _B
		if run_callbacks and C.onchange is not _A:
			try:C.onchange()
			except Exception as F:errors.display(F,f"changing setting {A} to {D}");setattr(B,A,E);return _B
		return _C
	def get_default(B,key):
		'returns the default value for the key';A=B.data_labels.get(key)
		if A is _A:return
		return A.default
	def save(A,filename):
		assert not cmd_opts.freeze_settings,'saving settings is disabled'
		with open(filename,'w',encoding='utf8')as B:json.dump(A.data,B,indent=4)
	def same_type(A,x,y):
		if x is _A or y is _A:return _C
		B=A.typemap.get(type(x),type(x));C=A.typemap.get(type(y),type(y));return B==C
	def load(A,filename):
		E='ui_reorder_list';F='quicksettings_list';G='quicksettings';H='sd_vae_overrides_per_model_preferences';I='sd_vae_as_default';J=filename;B='ui_reorder'
		with open(J,'r',encoding='utf8')as M:A.data=json.load(M)
		if A.data.get(I)is not _A and A.data.get(H)is _A:A.data[H]=not A.data.get(I)
		if A.data.get(G)is not _A and A.data.get(F)is _A:A.data[F]=[A.strip()for A in A.data.get(G).split(',')]
		if isinstance(A.data.get(B),str)and A.data.get(B)and E not in A.data:A.data[E]=[A.strip()for A in A.data.get(B).split(',')]
		K=0
		for(L,C)in A.data.items():
			D=A.data_labels.get(L,_A)
			if D is not _A and not A.same_type(D.default,C):print(f"Warning: bad setting value: {L}: {C} ({type(C).__name__}; expected {type(D.default).__name__})",file=sys.stderr);K+=1
		if K>0:print(f"The program is likely to not work with bad settings.\nSettings file: {J}\nEither fix the file, or delete it and restart.",file=sys.stderr)
	def onchange(A,key,func,call=_C):
		B=A.data_labels.get(key);B.onchange=func
		if call:func()
	def dumpjson(A):B={B:A.data.get(B,C.default)for(B,C)in A.data_labels.items()};B['_comments_before']={B:A.comment_before for(B,A)in A.data_labels.items()if A.comment_before is not _A};B['_comments_after']={B:A.comment_after for(B,A)in A.data_labels.items()if A.comment_after is not _A};return json.dumps(B)
	def add_option(A,key,info):A.data_labels[key]=info
	def reorder(B):
		'reorder settings so that all items related to section always go together';A={};C=B.data_labels.items()
		for(E,D)in C:
			if D.section not in A:A[D.section]=len(A)
		B.data_labels=dict(sorted(C,key=lambda x:A[x[1].section]))
	def cast_value(C,key,value):
		'casts an arbitrary to the same type as this setting\'s value with key\n        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)\n        ';A=value
		if A is _A:return
		B=C.data_labels[key].default
		if B is _A:B=getattr(C,key,_A)
		if B is _A:return
		D=type(B)
		if D==bool and A=='False':A=_B
		else:A=D(A)
		return A