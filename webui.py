from __future__ import annotations
import os,time
from modules import timer
from modules import initialize_util
from modules import initialize
startup_timer=timer.startup_timer
startup_timer.record('launcher')
initialize.imports()
initialize.check_versions()
def create_api(app):from modules.api.api import Api;from modules.call_queue import queue_lock as A;B=Api(app,A);return B
def api_only():from fastapi import FastAPI as D;from modules.shared_cmd_options import cmd_opts as A;initialize.initialize();B=D();initialize_util.setup_middleware(B);E=create_api(B);from modules import script_callbacks as C;C.before_ui_callback();C.app_started_callback(None,B);print(f"Startup time: {startup_timer.summary()}.");E.launch(server_name='0.0.0.0'if A.listen else'127.0.0.1',port=A.port if A.port else 7861,root_path=f"/{A.subpath}"if A.subpath else'')
def webui():
	I='SD_WEBUI_RESTARTING';H='stop';F=True;from modules.shared_cmd_options import cmd_opts as A;J=A.api;initialize.initialize();from modules import shared as B,ui_tempdir as K,script_callbacks as E,ui,progress as L,ui_extra_networks as M
	while 1:
		if B.opts.clean_temp_dir_at_start:K.cleanup_tmpdr();startup_timer.record('cleanup temp dir')
		E.before_ui_callback();startup_timer.record('scripts before_ui_callback');B.demo=ui.create_ui();startup_timer.record('create ui')
		if not A.no_gradio_queue:B.demo.queue(64)
		N=list(initialize_util.get_gradio_auth_creds())or None;G=False
		if os.getenv(I)!='1':
			if B.opts.auto_launch_browser=='Remote'or A.autolaunch:G=F
			elif B.opts.auto_launch_browser=='Local':G=not any([A.listen,A.share,A.ngrok,A.server_name])
		C,O,P=B.demo.launch(share=A.share,server_name=initialize_util.gradio_server_name(),server_port=A.port,ssl_keyfile=A.tls_keyfile,ssl_certfile=A.tls_certfile,ssl_verify=A.disable_tls_verify,debug=A.gradio_debug,auth=N,inbrowser=G,prevent_thread_lock=F,allowed_paths=A.gradio_allowed_path,app_kwargs={'docs_url':'/docs','redoc_url':'/redoc'},root_path=f"/{A.subpath}"if A.subpath else'');startup_timer.record('gradio launch');C.user_middleware=[A for A in C.user_middleware if A.cls.__name__!='CORSMiddleware'];initialize_util.setup_middleware(C);L.setup_progress_api(C);ui.setup_ui_api(C)
		if J:create_api(C)
		M.add_pages_to_demo(C);startup_timer.record('add APIs')
		with startup_timer.subcategory('app_started_callback'):E.app_started_callback(B.demo,C)
		timer.startup_record=startup_timer.dump();print(f"Startup time: {startup_timer.summary()}.")
		try:
			while F:
				D=B.state.wait_for_server_command(timeout=5)
				if D:
					if D in(H,'restart'):break
					else:print(f"Unknown server command: {D}")
		except KeyboardInterrupt:print('Caught KeyboardInterrupt, stopping...');D=H
		if D==H:print('Stopping server...');B.demo.close();break
		os.environ.setdefault(I,'1');print('Restarting UI...');B.demo.close();time.sleep(.5);startup_timer.reset();E.app_reload_callback();startup_timer.record('app reload callback');E.script_unloaded_callback();startup_timer.record('scripts unloaded callback');initialize.initialize_rest(reload_script_modules=F)
if __name__=='__main__':
	from modules.shared_cmd_options import cmd_opts
	if cmd_opts.nowebui:api_only()
	else:webui()