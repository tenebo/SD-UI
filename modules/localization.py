import json,os
from modules import errors,scripts
localizations={}
def list_localizations(dirname):
	E='.json';C=dirname;localizations.clear()
	for A in os.listdir(C):
		B,D=os.path.splitext(A)
		if D.lower()!=E:continue
		localizations[B]=os.path.join(C,A)
	for A in scripts.list_scripts('localizations',E):B,D=os.path.splitext(A.filename);localizations[B]=A.path
def localization_js(current_localization_name):
	A=localizations.get(current_localization_name,None);B={}
	if A is not None:
		try:
			with open(A,'r',encoding='utf8')as C:B=json.load(C)
		except Exception:errors.report(f"Error loading localization from {A}",exc_info=True)
	return f"window.localization = {json.dumps(B)}"