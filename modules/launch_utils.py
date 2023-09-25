_G='GRADIO_ANALYTICS_ENABLED'
_F='utf-8'
_E='<none>'
_D='utf8'
_C=False
_B=True
_A=None
import logging,re,subprocess,os,shutil,sys,importlib.util,platform,json
from functools import lru_cache
from modules import cmd_args,errors
from modules.paths_internal import script_path,extensions_dir
from modules.timer import startup_timer
from modules import logging_config
args,_=cmd_args.parser.parse_known_args()
logging_config.setup_logging(args.loglevel)
python=sys.executable
git=os.environ.get('GIT','git')
index_url=os.environ.get('INDEX_URL','')
dir_repos='repositories'
default_command_live=os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT')=='1'
if _G not in os.environ:os.environ[_G]='False'
def check_python_version():
	A=platform.system()=='Windows';B=sys.version_info.major;C=sys.version_info.minor;E=sys.version_info.micro
	if A:D=[10]
	else:D=[7,8,9,10,11]
	if not(B==3 and C in D):import modules.errors;modules.errors.print_error_explanation(f'''
INCOMPATIBLE PYTHON VERSION

This program is tested with 3.10.6 Python, but you have {B}.{C}.{E}.
If you encounter an error with "RuntimeError: Couldn\'t install torch." message,
or any other error regarding unsuccessful package (library) installation,
please downgrade (or upgrade) to the latest version of 3.10 Python
and delete current Python and "venv" folder in WebUI\'s directory.

You can download 3.10 Python from here: https://www.python.org/downloads/release/python-3106/

{"Alternatively, use a binary release of WebUI: https://github.com/tenebo/standard-demo-ourui/releases"if A else""}

Use --skip-python-version-check to suppress this warning.
''')
@lru_cache()
def commit_hash():
	try:return subprocess.check_output([git,'rev-parse','HEAD'],shell=_C,encoding=_D).strip()
	except Exception:return _E
@lru_cache()
def git_tag():
	try:return subprocess.check_output([git,'describe','--tags'],shell=_C,encoding=_D).strip()
	except Exception:
		try:
			B=os.path.join(os.path.dirname(os.path.dirname(__file__)),'CHANGELOG.md')
			with open(B,'r',encoding=_F)as C:A=next((A.strip()for A in C if A.strip()),_E);A=A.replace('## ','');return A
		except Exception:return _E
def run(command,desc=_A,errdesc=_A,custom_env=_A,live=default_command_live):
	E=custom_env;D=command
	if desc is not _A:print(desc)
	B={'args':D,'shell':_B,'env':os.environ if E is _A else E,'encoding':_D,'errors':'ignore'}
	if not live:B['stdout']=B['stderr']=subprocess.PIPE
	A=subprocess.run(**B)
	if A.returncode!=0:
		C=[f"{errdesc or'Error running command'}.",f"Command: {D}",f"Error code: {A.returncode}"]
		if A.stdout:C.append(f"stdout: {A.stdout}")
		if A.stderr:C.append(f"stderr: {A.stderr}")
		raise RuntimeError('\n'.join(C))
	return A.stdout or''
def is_installed(package):
	try:A=importlib.util.find_spec(package)
	except ModuleNotFoundError:return _C
	return A is not _A
def repo_dir(name):return os.path.join(script_path,dir_repos,name)
def run_pip(command,desc=_A,live=default_command_live):
	if args.skip_install:return
	A=f" --index-url {index_url}"if index_url!=''else'';return run(f'"{python}" -m pip {command} --prefer-binary{A}',desc=f"Installing {desc}",errdesc=f"Couldn't install {desc}",live=live)
def check_run_python(code):A=subprocess.run([python,'-c',code],capture_output=_B,shell=_C);return A.returncode==0
def git_fix_workspace(dir,name):A=name;run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc',f"Fetching all contents for {A}",f"Couldn't fetch {A}",live=_B);run(f'"{git}" -C "{dir}" gc --aggressive --prune=now',f"Pruning {A}",f"Couldn't prune {A}",live=_B)
def run_git(dir,name,command,desc=_A,errdesc=_A,custom_env=_A,live=default_command_live,autofix=_B):
	C=custom_env;B=command;A=errdesc
	try:return run(f'"{git}" -C "{dir}" {B}',desc=desc,errdesc=A,custom_env=C,live=live)
	except RuntimeError:
		if not autofix:raise
	print(f"{A}, attempting autofix...");git_fix_workspace(dir,name);return run(f'"{git}" -C "{dir}" {B}',desc=desc,errdesc=A,custom_env=C,live=live)
def git_clone(url,dir,name,commithash=_A):
	C=url;B=commithash;A=name
	if os.path.exists(dir):
		if B is _A:return
		D=run_git(dir,A,'rev-parse HEAD',_A,f"Couldn't determine {A}'s hash: {B}",live=_C).strip()
		if D==B:return
		if run_git(dir,A,'config --get remote.origin.url',_A,f"Couldn't determine {A}'s origin URL",live=_C).strip()!=C:run_git(dir,A,f'remote set-url origin "{C}"',_A,f"Failed to set {A}'s origin URL",live=_C)
		run_git(dir,A,'fetch',f"Fetching updates for {A}...",f"Couldn't fetch {A}",autofix=_C);run_git(dir,A,f"checkout {B}",f"Checking out commit for {A} with hash: {B}...",f"Couldn't checkout commit {B} for {A}",live=_B);return
	try:run(f'"{git}" clone "{C}" "{dir}"',f"Cloning {A} into {dir}...",f"Couldn't clone {A}",live=_B)
	except RuntimeError:shutil.rmtree(dir,ignore_errors=_B);raise
	if B is not _A:run(f'"{git}" -C "{dir}" checkout {B}',_A,"Couldn't checkout {name}'s hash: {commithash}")
def git_pull_recursive(dir):
	for(A,B,B)in os.walk(dir):
		if os.path.exists(os.path.join(A,'.git')):
			try:C=subprocess.check_output([git,'-C',A,'pull','--autostash']);print(f"Pulled changes for repository in '{A}':\n{C.decode(_F).strip()}\n")
			except subprocess.CalledProcessError as D:print(f"Couldn't perform 'git pull' on repository in '{A}':\n{D.output.decode(_F).strip()}\n")
def version_check(commit):
	E='--------------------------------------------------------';D='sha';C='commit';A=commit
	try:
		import requests as F;B=F.get('https://api.github.com/repos/tenebo/standard-demo-ourui/branches/master').json()
		if A!=_E and B[C][D]!=A:print(E);print('| You are not up to date with the most recent release. |');print('| Consider running `git pull` to update.               |');print(E)
		elif B[C][D]==A:print('You are up to date with the most recent release.')
		else:print("Not a git clone, can't perform version check.")
	except Exception as G:print('version check failed',G)
def run_extension_installer(extension_dir):
	E='PYTHONPATH';B=extension_dir;C=os.path.join(B,'install.py')
	if not os.path.isfile(C):return
	try:
		A=os.environ.copy();A[E]=f"{os.path.abspath('.')}{os.pathsep}{A.get(E,'')}";D=run(f'"{python}" "{C}"',errdesc=f"Error running install.py for extension {B}",custom_env=A).strip()
		if D:print(D)
	except Exception as F:errors.report(str(F))
def list_extensions(settings_file):
	C='none';B=settings_file;A={}
	try:
		if os.path.isfile(B):
			with open(B,'r',encoding=_D)as D:A=json.load(D)
	except Exception:errors.report('Could not load settings',exc_info=_B)
	E=set(A.get('disabled_extensions',[]));F=A.get('disable_all_extensions',C)
	if F!=C or args.disable_extra_extensions or args.disable_all_extensions or not os.path.isdir(extensions_dir):return[]
	return[A for A in os.listdir(extensions_dir)if A not in E]
def run_extensions_installers(settings_file):
	if not os.path.isdir(extensions_dir):return
	with startup_timer.subcategory('run extensions installers'):
		for A in list_extensions(settings_file):
			logging.debug(f"Installing {A}");B=os.path.join(extensions_dir,A)
			if os.path.isdir(B):run_extension_installer(B);startup_timer.record(A)
re_requirement=re.compile('\\s*([-_a-zA-Z0-9]+)\\s*(?:==\\s*([-+_.a-zA-Z0-9]+))?\\s*')
def requirements_met(requirements_file):
	'\n    Does a simple parse of a requirements.txt file to determine if all rerqirements in it\n    are already installed. Returns True if so, False if not installed or parsing fails.\n    ';import importlib.metadata,packaging.version
	with open(requirements_file,'r',encoding=_D)as D:
		for B in D:
			if B.strip()=='':continue
			A=re.match(re_requirement,B)
			if A is _A:return _C
			E=A.group(1).strip();C=(A.group(2)or'').strip()
			if C=='':continue
			try:F=importlib.metadata.version(E)
			except Exception:return _C
			if packaging.version.parse(C)!=packaging.version.parse(F):return _C
	return _B
def prepare_environment():
	I='BLIP';H='install ngrok';G='ngrok';F='xformers';E='open_clip';D='clip';B='CodeFormer';J=os.environ.get('TORCH_INDEX_URL','https://download.pytorch.org/whl/cu118');K=os.environ.get('TORCH_COMMAND',f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {J}");A=os.environ.get('REQS_FILE','requirements_versions.txt');L=os.environ.get('XFORMERS_PACKAGE','xformers==0.0.20');M=os.environ.get('CLIP_PACKAGE','https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip');N=os.environ.get('OPENCLIP_PACKAGE','https://github.com/mlfoundations/open_clip/archive/bb6e834e9c70d9c27d0dc3ecedeebeaeb1ffad6b.zip');O=os.environ.get('STABLE_DIFFUSION_REPO','https://github.com/Stability-AI/stablediffusion.git');P=os.environ.get('STABLE_DIFFUSION_XL_REPO','https://github.com/Stability-AI/generative-models.git');Q=os.environ.get('K_DIFFUSION_REPO','https://github.com/crowsonkb/k-diffusion.git');R=os.environ.get('CODEFORMER_REPO','https://github.com/sczhou/CodeFormer.git');S=os.environ.get('BLIP_REPO','https://github.com/salesforce/BLIP.git');T=os.environ.get('STABLE_DIFFUSION_COMMIT_HASH','cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf');U=os.environ.get('STABLE_DIFFUSION_XL_COMMIT_HASH','45c443b316737a4ab6e40413d7794a7f5657c19f');V=os.environ.get('K_DIFFUSION_COMMIT_HASH','ab527a9a6d347f364e3d185ba6d714e22d80cb3c');W=os.environ.get('CODEFORMER_COMMIT_HASH','c5b4593074ba6214284d6acd5f1719b6c5d739af');X=os.environ.get('BLIP_COMMIT_HASH','48211a1594f1321b00f14c9f7a5b4813144b2fb9')
	try:os.remove(os.path.join(script_path,'tmp','restart'));os.environ.setdefault('SD_WEBUI_RESTARTING','1')
	except OSError:pass
	if not args.skip_python_version_check:check_python_version()
	startup_timer.record('checks');C=commit_hash();Y=git_tag();startup_timer.record('git version info');print(f"Python {sys.version}");print(f"Version: {Y}");print(f"Commit hash: {C}")
	if args.reinstall_torch or not is_installed('torch')or not is_installed('torchvision'):run(f'"{python}" -m {K}','Installing torch and torchvision',"Couldn't install torch",live=_B);startup_timer.record('install torch')
	if not args.skip_torch_cuda_test and not check_run_python('import torch; assert torch.cuda.is_available()'):raise RuntimeError('Torch is not able to use GPU; add --skip-torch-cuda-test to COMMANDLINE_ARGS variable to disable this check')
	startup_timer.record('torch GPU test')
	if not is_installed(D):run_pip(f"install {M}",D);startup_timer.record('install clip')
	if not is_installed(E):run_pip(f"install {N}",E);startup_timer.record('install open_clip')
	if(not is_installed(F)or args.reinstall_xformers)and args.xformers:run_pip(f"install -U -I --no-deps {L}",F);startup_timer.record('install xformers')
	if not is_installed(G)and args.ngrok:run_pip(H,G);startup_timer.record(H)
	os.makedirs(os.path.join(script_path,dir_repos),exist_ok=_B);git_clone(O,repo_dir('stable-diffusion-stability-ai'),'Stable Diffusion',T);git_clone(P,repo_dir('generative-models'),'Stable Diffusion XL',U);git_clone(Q,repo_dir('k-diffusion'),'K-diffusion',V);git_clone(R,repo_dir(B),B,W);git_clone(S,repo_dir(I),I,X);startup_timer.record('clone repositores')
	if not is_installed('lpips'):run_pip(f'install -r "{os.path.join(repo_dir(B),"requirements.txt")}"','requirements for CodeFormer');startup_timer.record('install CodeFormer requirements')
	if not os.path.isfile(A):A=os.path.join(script_path,A)
	if not requirements_met(A):run_pip(f'install -r "{A}"','requirements');startup_timer.record('install requirements')
	if not args.skip_install:run_extensions_installers(settings_file=args.ui_settings_file)
	if args.update_check:version_check(C);startup_timer.record('check version')
	if args.update_all_extensions:git_pull_recursive(extensions_dir);startup_timer.record('update extensions')
	if'--exit'in sys.argv:print('Exiting because of --exit argument');exit(0)
def configure_for_tests():
	D='--disable-nan-check';C='--skip-torch-cuda-test';B='--ckpt';A='--api'
	if A not in sys.argv:sys.argv.append(A)
	if B not in sys.argv:sys.argv.append(B);sys.argv.append(os.path.join(script_path,'test/test_files/empty.pt'))
	if C not in sys.argv:sys.argv.append(C)
	if D not in sys.argv:sys.argv.append(D)
	os.environ['COMMANDLINE_ARGS']=''
def start():
	B='--nowebui';print(f"Launching {'API server'if B in sys.argv else'Web UI'} with arguments: {' '.join(sys.argv[1:])}");import webui as A
	if B in sys.argv:A.api_only()
	else:A.webui()
def dump_sysinfo():
	from modules import sysinfo as B;import datetime as C;D=B.get();A=f"sysinfo-{C.datetime.utcnow().strftime('%Y-%m-%d-%H-%M')}.txt"
	with open(A,'w',encoding=_D)as E:E.write(D)
	return A