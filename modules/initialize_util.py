import json,os,signal,sys,re
from modules.timer import startup_timer
def gradio_server_name():
	from modules.shared_cmd_options import cmd_opts as A
	if A.server_name:return A.server_name
	else:return'0.0.0.0'if A.listen else None
def fix_torch_version():
	import torch as A
	if'.dev'in A.__version__ or'+git'in A.__version__:A.__long_version__=A.__version__;A.__version__=re.search('[\\d.]+[\\d]',A.__version__).group(0)
def fix_asyncio_event_loop_policy():
	'\n        The default `asyncio` event loop policy only automatically creates\n        event loops in the main threads. Other threads must create event\n        loops explicitly or `asyncio.get_event_loop` (and therefore\n        `.IOLoop.current`) will fail. Installing this policy allows event\n        loops to be created automatically on any thread, matching the\n        behavior of Tornado versions prior to 5.0 (or 5.0 on Python 2).\n    ';import asyncio as A
	if sys.platform=='win32'and hasattr(A,'WindowsSelectorEventLoopPolicy'):B=A.WindowsSelectorEventLoopPolicy
	else:B=A.DefaultEventLoopPolicy
	class C(B):
		'Event loop policy that allows loop creation on any thread.\n        Usage::\n\n            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())\n        '
		def get_event_loop(A):
			try:return super().get_event_loop()
			except(RuntimeError,AssertionError):B=A.new_event_loop();A.set_event_loop(B);return B
	A.set_event_loop_policy(C())
def restore_config_state_file():
	from modules import shared as B,config_states as C;A=B.opts.restore_config_state_file
	if A=='':return
	B.opts.restore_config_state_file='';B.opts.save(B.config_filename)
	if os.path.isfile(A):
		print(f"*** About to restore extension state from file: {A}")
		with open(A,'r',encoding='utf-8')as D:E=json.load(D);C.restore_extension_config(E)
		startup_timer.record('restore extension config')
	elif A:print(f"!!! Config state backup not found: {A}")
def validate_tls_options():
	from modules.shared_cmd_options import cmd_opts as A
	if not(A.tls_keyfile and A.tls_certfile):return
	try:
		if not os.path.exists(A.tls_keyfile):print('Invalid path to TLS keyfile given')
		if not os.path.exists(A.tls_certfile):print(f"Invalid path to TLS certfile: '{A.tls_certfile}'")
	except TypeError:A.tls_keyfile=A.tls_certfile=None;print('TLS setup invalid, running ourui without TLS')
	else:print('Running with TLS')
	startup_timer.record('TLS')
def get_gradio_auth_creds():
	'\n    Convert the gradio_auth and gradio_auth_path commandline arguments into\n    an iterable of (username, password) tuples.\n    ';from modules.shared_cmd_options import cmd_opts as B
	def C(s):
		s=s.strip()
		if not s:return
		return tuple(s.split(':',1))
	if B.gradio_auth:
		for A in B.gradio_auth.split(','):
			A=C(A)
			if A:yield A
	if B.gradio_auth_path:
		with open(B.gradio_auth_path,'r',encoding='utf8')as D:
			for E in D.readlines():
				for A in E.strip().split(','):
					A=C(A)
					if A:yield A
def dumpstacks():
	import threading as D,traceback as E;F={A.ident:A.name for A in D.enumerate()};A=[]
	for(B,G)in sys._current_frames().items():
		A.append(f"\n# Thread: {F.get(B,'')}({B})")
		for(H,I,J,C)in E.extract_stack(G):
			A.append(f'File: "{H}", line {I}, in {J}')
			if C:A.append('  '+C.strip())
	print('\n'.join(A))
def configure_sigint_handler():
	def A(sig,frame):print(f"Interrupted with signal {sig} in {frame}");dumpstacks();os._exit(0)
	if not os.environ.get('COVERAGE_RUN'):signal.signal(signal.SIGINT,A)
def configure_opts_onchange():B=False;from modules import shared as A,sd_models as E,sd_vae as D,ui_tempdir as F,sd_hijack as G;from modules.call_queue import wrap_queued_call as C;A.opts.onchange('sd_model_checkpoint',C(lambda:E.reload_model_weights()),call=B);A.opts.onchange('sd_vae',C(lambda:D.reload_vae_weights()),call=B);A.opts.onchange('sd_vae_overrides_per_model_preferences',C(lambda:D.reload_vae_weights()),call=B);A.opts.onchange('temp_dir',F.on_tmpdir_changed);A.opts.onchange('gradio_theme',A.reload_gradio_theme);A.opts.onchange('cross_attention_optimization',C(lambda:G.model_hijack.redo_hijack(A.sd_model)),call=B);startup_timer.record('opts onchange')
def setup_middleware(app):A=app;from starlette.middleware.gzip import GZipMiddleware as B;A.middleware_stack=None;A.add_middleware(B,minimum_size=1000);configure_cors_middleware(A);A.build_middleware_stack()
def configure_cors_middleware(app):
	from starlette.middleware.cors import CORSMiddleware as C;from modules.shared_cmd_options import cmd_opts as A;B={'allow_methods':['*'],'allow_headers':['*'],'allow_credentials':True}
	if A.cors_allow_origins:B['allow_origins']=A.cors_allow_origins.split(',')
	if A.cors_allow_origins_regex:B['allow_origin_regex']=A.cors_allow_origins_regex
	app.add_middleware(C,**B)