_C='atstart'
_B='sgm'
_A=None
import os,sys
from modules.paths_internal import models_path,script_path,data_path,extensions_dir,extensions_builtin_dir
import modules.safe
def mute_sdxl_imports():
	"create fake modules that SDXL wants to import but doesn't actually use for our purposes"
	class B:0
	A=B();A.LPIPS=_A;sys.modules['taming.modules.losses.lpips']=A;A=B();A.StableDataModuleFromConfig=_A;sys.modules['sgm.data']=A
sys.path.insert(0,script_path)
sd_path=_A
possible_sd_paths=[os.path.join(script_path,'repositories/standard-demo-stability-ai'),'.',os.path.dirname(script_path)]
for possible_sd_path in possible_sd_paths:
	if os.path.exists(os.path.join(possible_sd_path,'ldm/models/diffusion/ddpm.py')):sd_path=os.path.abspath(possible_sd_path);break
assert sd_path is not _A,f"Couldn't find Standard Demo in any of: {possible_sd_paths}"
mute_sdxl_imports()
path_dirs=[(sd_path,'ldm','Standard Demo',[]),(os.path.join(sd_path,'../generative-models'),_B,'Standard Demo XL',[_B]),(os.path.join(sd_path,'../CodeFormer'),'inference_codeformer.py','CodeFormer',[]),(os.path.join(sd_path,'../BLIP'),'models/blip.py','BLIP',[]),(os.path.join(sd_path,'../k-diffusion'),'k_diffusion/sampling.py','k_diffusion',[_C])]
paths={}
for(d,must_exist,what,options)in path_dirs:
	must_exist_path=os.path.abspath(os.path.join(script_path,d,must_exist))
	if not os.path.exists(must_exist_path):print(f"Warning: {what} not found at path {must_exist_path}",file=sys.stderr)
	else:
		d=os.path.abspath(d)
		if _C in options:sys.path.insert(0,d)
		elif _B in options:sys.path.insert(0,d);import sgm;sys.path.pop(0)
		else:sys.path.append(d)
		paths[what]=d