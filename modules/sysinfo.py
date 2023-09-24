_A=False
import json,os,sys,traceback,platform,hashlib,pkg_resources,psutil,re,launch
from modules import paths_internal,timer,shared,extensions,errors
checksum_token='DontStealMyGamePlz__WINNERS_DONT_USE_DRUGS__DONT_COPY_THAT_FLOPPY'
environment_whitelist={'GIT','INDEX_URL','WEBUI_LAUNCH_LIVE_OUTPUT','GRADIO_ANALYTICS_ENABLED','PYTHONPATH','TORCH_INDEX_URL','TORCH_COMMAND','REQS_FILE','XFORMERS_PACKAGE','CLIP_PACKAGE','OPENCLIP_PACKAGE','STABLE_DIFFUSION_REPO','K_DIFFUSION_REPO','CODEFORMER_REPO','BLIP_REPO','STABLE_DIFFUSION_COMMIT_HASH','K_DIFFUSION_COMMIT_HASH','CODEFORMER_COMMIT_HASH','BLIP_COMMIT_HASH','COMMANDLINE_ARGS','IGNORE_CMD_ARGS_ERRORS'}
def pretty_bytes(num,suffix='B'):
	A=num
	for B in['','K','M','G','T','P','E','Z','Y']:
		if abs(A)<1024 or B=='Y':return f"{A:.0f}{B}{suffix}"
		A/=1024
def get():B=get_dict();A=json.dumps(B,ensure_ascii=_A,indent=4);C=hashlib.sha256(A.encode('utf8'));A=A.replace(checksum_token,C.hexdigest());return A
re_checksum=re.compile('"Checksum": "([0-9a-fA-F]{64})"')
def check(x):
	A=re.search(re_checksum,x)
	if not A:return _A
	B=re.sub(re_checksum,f'"Checksum": "{checksum_token}"',x);C=hashlib.sha256(B.encode('utf8'));return C.hexdigest()==A.group(1)
def get_dict():B=psutil.virtual_memory();A={'Platform':platform.platform(),'Python':platform.python_version(),'Version':launch.git_tag(),'Commit':launch.commit_hash(),'Script path':paths_internal.script_path,'Data path':paths_internal.data_path,'Extensions dir':paths_internal.extensions_dir,'Checksum':checksum_token,'Commandline':get_argv(),'Torch env info':get_torch_sysinfo(),'Exceptions':get_exceptions(),'CPU':{'model':platform.processor(),'count logical':psutil.cpu_count(logical=True),'count physical':psutil.cpu_count(logical=_A)},'RAM':{A:pretty_bytes(getattr(B,A,0))for A in['total','used','free','active','inactive','buffers','cached','shared']if getattr(B,A,0)!=0},'Extensions':get_extensions(enabled=True),'Inactive extensions':get_extensions(enabled=_A),'Environment':get_environment(),'Config':get_config(),'Startup':timer.startup_record,'Packages':sorted([f"{A.key}=={A.version}"for A in pkg_resources.working_set])};return A
def format_traceback(tb):return[[f"{A.filename}, line {A.lineno}, {A.name}",A.line]for A in traceback.extract_tb(tb)]
def format_exception(e,tb):return{'exception':str(e),'traceback':format_traceback(tb)}
def get_exceptions():
	try:return list(reversed(errors.exception_records))
	except Exception as A:return str(A)
def get_environment():return{A:os.environ[A]for A in sorted(os.environ)if A in environment_whitelist}
def get_argv():
	C='<hidden>';A=[]
	for B in sys.argv:
		if shared.cmd_opts.gradio_auth and shared.cmd_opts.gradio_auth==B:A.append(C);continue
		if shared.cmd_opts.api_auth and shared.cmd_opts.api_auth==B:A.append(C);continue
		A.append(B)
	return A
re_newline=re.compile('\\r*\\n')
def get_torch_sysinfo():
	try:import torch.utils.collect_env;A=torch.utils.collect_env.get_env_info()._asdict();return{B:re.split(re_newline,str(A))if'\n'in str(A)else A for(B,A)in A.items()}
	except Exception as B:return str(B)
def get_extensions(*,enabled):
	try:
		def B(x):return{'name':x.name,'path':x.path,'version':x.version,'branch':x.branch,'remote':x.remote}
		return[B(A)for A in extensions.extensions if not A.is_builtin and A.enabled==enabled]
	except Exception as A:return str(A)
def get_config():
	try:return shared.opts.data
	except Exception as A:return str(A)