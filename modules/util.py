import os,re
from modules import shared
from modules.paths_internal import script_path
def natural_sort_key(s,regex=re.compile('([0-9]+)')):return[int(A)if A.isdigit()else A.lower()for A in regex.split(s)]
def listfiles(dirname):A=dirname;B=[os.path.join(A,B)for B in sorted(os.listdir(A),key=natural_sort_key)if not B.startswith('.')];return[A for A in B if os.path.isfile(A)]
def html_path(filename):return os.path.join(script_path,'html',filename)
def html(filename):
	A=html_path(filename)
	if os.path.exists(A):
		with open(A,encoding='utf8')as B:return B.read()
	return''
def walk_files(path,allowed_extensions=None):
	A=allowed_extensions
	if not os.path.exists(path):return
	if A is not None:A=set(A)
	B=list(os.walk(path,followlinks=True));B=sorted(B,key=lambda x:natural_sort_key(x[0]))
	for(C,E,F)in B:
		for D in sorted(F,key=natural_sort_key):
			if A is not None:
				E,G=os.path.splitext(D)
				if G not in A:continue
			if not shared.opts.list_hidden_files and('/.'in C or'\\.'in C):continue
			yield os.path.join(C,D)
def ldm_print(*A,**B):
	if shared.opts.hide_ldm_prints:return
	print(*A,**B)