_D='origin'
_C=False
_B=None
_A=True
import os,threading
from modules import shared,errors,cache,scripts
from modules.gitpython_hack import Repo
from modules.paths_internal import extensions_dir,extensions_builtin_dir,script_path
extensions=[]
os.makedirs(extensions_dir,exist_ok=_A)
def active():
	if shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions=='all':return[]
	elif shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions=='extra':return[A for A in extensions if A.enabled and A.is_builtin]
	else:return[A for A in extensions if A.enabled]
class Extension:
	lock=threading.Lock();cached_fields=['remote','commit_date','branch','commit_hash','version']
	def __init__(A,name,path,enabled=_A,is_builtin=_C):A.name=name;A.path=path;A.enabled=enabled;A.status='';A.can_update=_C;A.is_builtin=is_builtin;A.commit_hash='';A.commit_date=_B;A.version='';A.branch=_B;A.remote=_B;A.have_info_from_repo=_C
	def to_dict(A):return{B:getattr(A,B)for B in A.cached_fields}
	def from_dict(A,d):
		for B in A.cached_fields:setattr(A,B,d[B])
	def read_info_from_repo(A):
		if A.is_builtin or A.have_info_from_repo:return
		def B():
			with A.lock:
				if A.have_info_from_repo:return
				A.do_read_info_from_repo();return A.to_dict()
		try:C=cache.cached_data_for_file('extensions-git',A.name,os.path.join(A.path,'.git'),B);A.from_dict(C)
		except FileNotFoundError:pass
		A.status='unknown'if A.status==''else A.status
	def do_read_info_from_repo(A):
		B=_B
		try:
			if os.path.exists(os.path.join(A.path,'.git')):B=Repo(A.path)
		except Exception:errors.report(f"Error reading github repository info from {A.path}",exc_info=_A)
		if B is _B or B.bare:A.remote=_B
		else:
			try:
				A.remote=next(B.remote().urls,_B);C=B.head.commit;A.commit_date=C.committed_date
				if B.active_branch:A.branch=B.active_branch.name
				A.commit_hash=C.hexsha;A.version=A.commit_hash[:8]
			except Exception:errors.report(f"Failed reading extension data from Git repository ({A.name})",exc_info=_A);A.remote=_B
		A.have_info_from_repo=_A
	def list_files(C,subdir,extension):
		B=os.path.join(C.path,subdir)
		if not os.path.isdir(B):return[]
		A=[]
		for D in sorted(os.listdir(B)):A.append(scripts.ScriptFile(C.path,D,os.path.join(B,D)))
		A=[A for A in A if os.path.splitext(A.path)[1].lower()==extension and os.path.isfile(A.path)];return A
	def check_updates(A):
		B=Repo(A.path)
		for C in B.remote().fetch(dry_run=_A):
			if C.flags!=C.HEAD_UPTODATE:A.can_update=_A;A.status='new commits';return
		try:
			D=B.rev_parse(_D)
			if B.head.commit!=D:A.can_update=_A;A.status='behind HEAD';return
		except Exception:A.can_update=_C;A.status='unknown (remote error)';return
		A.can_update=_C;A.status='latest'
	def fetch_and_reset_hard(A,commit=_D):B=Repo(A.path);B.git.fetch(all=_A);B.git.reset(commit,hard=_A);A.have_info_from_repo=_C
def list_extensions():
	extensions.clear()
	if not os.path.isdir(extensions_dir):return
	if shared.cmd_opts.disable_all_extensions:print('*** "--disable-all-extensions" arg was used, will not load any extensions ***')
	elif shared.opts.disable_all_extensions=='all':print('*** "Disable all extensions" option was set, will not load any extensions ***')
	elif shared.cmd_opts.disable_extra_extensions:print('*** "--disable-extra-extensions" arg was used, will only load built-in extensions ***')
	elif shared.opts.disable_all_extensions=='extra':print('*** "Disable all extensions" option was set, will only load built-in extensions ***')
	C=[]
	for A in[extensions_dir,extensions_builtin_dir]:
		if not os.path.isdir(A):return
		for D in sorted(os.listdir(A)):
			B=os.path.join(A,D)
			if not os.path.isdir(B):continue
			C.append((D,B,A==extensions_builtin_dir))
	for(A,B,E)in C:F=Extension(name=A,path=B,enabled=A not in shared.opts.disabled_extensions,is_builtin=E);extensions.append(F)