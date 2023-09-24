'\nSupports saving and restoring webui and extensions from a known working set of commits\n'
_J='enabled'
_I='branch'
_H='commit_date'
_G='remote'
_F='extensions'
_E='webui'
_D='created_at'
_C='commit_hash'
_B=None
_A=True
import os,json,time,tqdm
from datetime import datetime
import git
from modules import shared,extensions,errors
from modules.paths_internal import script_path,config_states_dir
all_config_states={}
def list_config_states():
	global all_config_states;all_config_states.clear();os.makedirs(config_states_dir,exist_ok=_A);A=[]
	for E in os.listdir(config_states_dir):
		if E.endswith('.json'):
			B=os.path.join(config_states_dir,E)
			try:
				with open(B,'r',encoding='utf-8')as F:C=json.load(F);assert _D in C,'"created_at" does not exist';C['filepath']=B;A.append(C)
			except Exception as G:print(f"[ERROR]: Config states {B}, {G}")
	A=sorted(A,key=lambda cs:cs[_D],reverse=_A)
	for D in A:H=time.asctime(time.gmtime(D[_D]));I=D.get('name','Config');J=f"{I}: {H}";all_config_states[J]=D
	return all_config_states
def get_webui_config():
	A=_B
	try:
		if os.path.exists(os.path.join(script_path,'.git')):A=git.Repo(script_path)
	except Exception:errors.report(f"Error reading webui git info from {script_path}",exc_info=_A)
	B=_B;C=_B;D=_B;E=_B
	if A and not A.bare:
		try:B=next(A.remote().urls,_B);F=A.head.commit;D=A.head.commit.committed_date;C=F.hexsha;E=A.active_branch.name
		except Exception:B=_B
	return{_G:B,_C:C,_H:D,_I:E}
def get_extension_config():
	B={}
	for A in extensions.extensions:A.read_info_from_repo();C={'name':A.name,'path':A.path,_J:A.enabled,'is_builtin':A.is_builtin,_G:A.remote,_C:A.commit_hash,_H:A.commit_date,_I:A.branch,'have_info_from_repo':A.have_info_from_repo};B[A.name]=C
	return B
def get_config():A=datetime.now().timestamp();B=get_webui_config();C=get_extension_config();return{_D:A,_E:B,_F:C}
def restore_webui_config(config):
	C=config;print('* Restoring webui state...')
	if _E not in C:print('Error: No webui data saved to config');return
	D=C[_E]
	if _C not in D:print('Error: No commit saved to webui config');return
	A=D.get(_C,_B);B=_B
	try:
		if os.path.exists(os.path.join(script_path,'.git')):B=git.Repo(script_path)
	except Exception:errors.report(f"Error reading webui git info from {script_path}",exc_info=_A);return
	try:B.git.fetch(all=_A);B.git.reset(A,hard=_A);print(f"* Restored webui to commit {A}.")
	except Exception:errors.report(f"Error restoring webui to commit{A}")
def restore_extension_config(config):
	G=config;C=False;print('* Restoring extension state...')
	if _F not in G:print('Error: No extension data saved to config');return
	H=G[_F];D=[];F=[]
	for A in tqdm.tqdm(extensions.extensions):
		if A.is_builtin:continue
		A.read_info_from_repo();E=A.commit_hash
		if A.name not in H:A.disabled=_A;F.append(A.name);D.append((A,E[:8],C,'Saved extension state not found in config, marking as disabled'));continue
		B=H[A.name]
		if _C in B and B[_C]:
			try:
				A.fetch_and_reset_hard(B[_C]);A.read_info_from_repo()
				if E!=B[_C]:D.append((A,E[:8],_A,B[_C][:8]))
			except Exception as J:D.append((A,E[:8],C,J))
		else:D.append((A,E[:8],C,'No commit hash found in config'))
		if not B.get(_J,C):A.disabled=_A;F.append(A.name)
		else:A.disabled=C
	shared.opts.disabled_extensions=F;shared.opts.save(shared.config_filename);print('* Finished restoring extensions. Results:')
	for(A,K,L,I)in D:
		if L:print(f"  + {A.name}: {K} -> {I}")
		else:print(f"  ! {A.name}: FAILURE ({I})")