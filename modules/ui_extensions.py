_S='Installed'
_R='Install'
_Q='commit_time'
_P='\n        </tbody>\n    </table>\n    '
_O='filepath'
_N='Config'
_M='stars'
_L='added'
_K='created_at'
_J='checked="checked"'
_I='both'
_H='installed'
_G='name'
_F='ourui'
_E='extensions'
_D='Current'
_C=True
_B=None
_A=False
import json,os,threading,time
from datetime import datetime,timezone
import git,gradio as gr,html,shutil,errno
from modules import extensions,shared,paths,config_states,errors,restart
from modules.paths_internal import config_states_dir
from modules.call_queue import wrap_gradio_gpu_call
available_extensions={_E:[]}
STYLE_PRIMARY=' style="color: var(--primary-400)"'
def check_access():assert not shared.cmd_opts.disable_extension_access,'extension access disabled because of command line flags'
def apply_and_restart(disable_list,update_list,disable_all):
	C=update_list;D=disable_list;check_access();E=json.loads(D);assert type(E)==list,f"wrong disable_list data for apply_and_restart: {D}";A=json.loads(C);assert type(A)==list,f"wrong update_list data for apply_and_restart: {C}"
	if A:save_config_state('Backup (pre-update)')
	A=set(A)
	for B in extensions.extensions:
		if B.name not in A:continue
		try:B.fetch_and_reset_hard()
		except Exception:errors.report(f"Error getting updates for {B.name}",exc_info=_C)
	shared.opts.disabled_extensions=E;shared.opts.disable_all_extensions=disable_all;shared.opts.save(shared.config_filename)
	if restart.is_restartable():restart.restart_program()
	else:restart.stop_program()
def save_config_state(name):
	A=name;C=config_states.get_config()
	if not A:A=_N
	C[_G]=A;D=datetime.now().strftime('%Y_%m_%d-%H_%M_%S');B=os.path.join(config_states_dir,f"{D}_{A}.json");print(f"Saving backup of ourui/extension state to {B}.")
	with open(B,'w',encoding='utf-8')as E:json.dump(C,E,indent=4)
	config_states.list_config_states();F=next(iter(config_states.all_config_states.keys()),_D);G=[_D]+list(config_states.all_config_states.keys());return gr.Dropdown.update(value=F,choices=G),f'<span>Saved current ourui/extension state to "{B}"</span>'
def restore_config_state(confirmed,config_state_name,restore_type):
	B=config_state_name;A=restore_type
	if B==_D:return'<span>Select a config to restore from.</span>'
	if not confirmed:return'<span>Cancelled.</span>'
	check_access();C=config_states.all_config_states[B];print(f"*** Restoring ourui state from backup: {A} ***")
	if A==_E or A==_I:shared.opts.restore_config_state_file=C[_O];shared.opts.save(shared.config_filename)
	if A==_F or A==_I:config_states.restore_ourui_config(C)
	shared.state.request_restart();return''
def check_updates(id_task,disable_list):
	B=disable_list;check_access();C=json.loads(B);assert type(C)==list,f"wrong disable_list data for apply_and_restart: {B}";D=[A for A in extensions.extensions if A.remote is not _B and A.name not in C];shared.state.job_count=len(D)
	for A in D:
		shared.state.textinfo=A.name
		try:A.check_updates()
		except FileNotFoundError as E:
			if'FETCH_HEAD'not in str(E):raise
		except Exception:errors.report(f"Error checking updates for {A.name}",exc_info=_C)
		shared.state.nextjob()
	return extension_table(),''
def make_commit_link(commit_hash,remote,text=_B):
	C=commit_hash;B=text;A=remote
	if B is _B:B=C[:8]
	if A.startswith('https://github.com/'):
		if A.endswith('.git'):A=A[:-4]
		D=A+'/commit/'+C;return f'<a href="{D}" target="_blank">{B}</a>'
	else:return B
def extension_table():
	B=f'''<!-- {time.time()} -->
    <table id="extensions">
        <thead>
            <tr>
                <th>
                    <input class="gr-check-radio gr-checkbox all_extensions_toggle" type="checkbox" {_J if all(A.enabled for A in extensions.extensions)else""} onchange="toggle_all_extensions(event)" />
                    <abbr title="Use checkbox to enable the extension; it will be enabled or disabled when you click apply button">Extension</abbr>
                </th>
                <th>URL</th>
                <th>Branch</th>
                <th>Version</th>
                <th>Date</th>
                <th><abbr title="Use checkbox to mark the extension for update; it will be updated when you click apply button">Update</abbr></th>
            </tr>
        </thead>
        <tbody>
    '''
	for A in extensions.extensions:
		A:0;A.read_info_from_repo();F=f'<a href="{html.escape(A.remote or"")}" target="_blank">{html.escape("built-in"if A.is_builtin else A.remote or"")}</a>'
		if A.can_update:C=f'<label><input class="gr-check-radio gr-checkbox" name="update_{html.escape(A.name)}" checked="checked" type="checkbox">{html.escape(A.status)}</label>'
		else:C=A.status
		D=''
		if shared.cmd_opts.disable_extra_extensions and not A.is_builtin or shared.opts.disable_all_extensions=='extra'and not A.is_builtin or shared.cmd_opts.disable_all_extensions or shared.opts.disable_all_extensions=='all':D=STYLE_PRIMARY
		E=A.version
		if A.commit_hash and A.remote:E=make_commit_link(A.commit_hash,A.remote,A.version)
		B+=f'''
            <tr>
                <td><label{D}><input class="gr-check-radio gr-checkbox extension_toggle" name="enable_{html.escape(A.name)}" type="checkbox" {_J if A.enabled else""} onchange="toggle_extension(event)" />{html.escape(A.name)}</label></td>
                <td>{F}</td>
                <td>{A.branch}</td>
                <td>{E}</td>
                <td>{datetime.fromtimestamp(A.commit_date)if A.commit_date else""}</td>
                <td{' class="extension_status"'if A.remote is not _B else""}>{C}</td>
            </tr>
    '''
	B+=_P;return B
def update_config_states_table(state_name):
	X='commit_date';Y=state_name;M='commit_hash';N='branch';O='remote';C='<unknown>'
	if Y==_D:A=config_states.get_config()
	else:A=config_states.all_config_states[Y]
	Z=A.get(_G,_N);a=time.asctime(time.gmtime(A[_K]));P=A.get(_O,C)
	try:
		D=A[_F][O]or'';b=A[_F][N];Q=A[_F][M]or C;E=A[_F][X]
		if E:E=time.asctime(time.gmtime(E))
		else:E=C
		R=f'<a href="{html.escape(D)}" target="_blank">{html.escape(D or"")}</a>';S=make_commit_link(Q,D);T=make_commit_link(Q,D,E);U=config_states.get_ourui_config();F='';G='';B=''
		if U[O]!=D:F=STYLE_PRIMARY
		if U[N]!=b:G=STYLE_PRIMARY
		if U[M]!=Q:B=STYLE_PRIMARY
		L=f'''<!-- {time.time()} -->
<h2>Config Backup: {Z}</h2>
<div><b>Filepath:</b> {P}</div>
<div><b>Created at:</b> {a}</div>
<h2>WebUI State</h2>
<table id="config_state_ourui">
    <thead>
        <tr>
            <th>URL</th>
            <th>Branch</th>
            <th>Commit</th>
            <th>Date</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
                <label{F}>{R}</label>
            </td>
            <td>
                <label{G}>{b}</label>
            </td>
            <td>
                <label{B}>{S}</label>
            </td>
            <td>
                <label{B}>{T}</label>
            </td>
        </tr>
    </tbody>
</table>
<h2>Extension State</h2>
<table id="config_state_extensions">
    <thead>
        <tr>
            <th>Extension</th>
            <th>URL</th>
            <th>Branch</th>
            <th>Commit</th>
            <th>Date</th>
        </tr>
    </thead>
    <tbody>
''';c={A.name:A for A in extensions.extensions}
		for(V,H)in A[_E].items():
			I=H[O]or'';d=H[N]or C;e=H['enabled'];W=H[M]or C;J=H[X]
			if J:J=time.asctime(time.gmtime(J))
			else:J=C
			R=f'<a href="{html.escape(I)}" target="_blank">{html.escape(I or"")}</a>';S=make_commit_link(W,I);T=make_commit_link(W,I,J);f='';F='';G='';B=''
			if V in c:
				K=c[V];K.read_info_from_repo()
				if K.enabled!=e:f=STYLE_PRIMARY
				if K.remote!=I:F=STYLE_PRIMARY
				if K.branch!=d:G=STYLE_PRIMARY
				if K.commit_hash!=W:B=STYLE_PRIMARY
			L+=f'''        <tr>
            <td><label{f}><input class="gr-check-radio gr-checkbox" type="checkbox" disabled="true" {_J if e else""}>{html.escape(V)}</label></td>
            <td><label{F}>{R}</label></td>
            <td><label{G}>{d}</label></td>
            <td><label{B}>{S}</label></td>
            <td><label{B}>{T}</label></td>
        </tr>
'''
		L+='    </tbody>\n</table>'
	except Exception as g:print(f"[ERROR]: Config states {P}, {g}");L=f"<!-- {time.time()} -->\n<h2>Config Backup: {Z}</h2>\n<div><b>Filepath:</b> {P}</div>\n<div><b>Created at:</b> {a}</div>\n<h2>This file is corrupted</h2>"
	return L
def normalize_git_url(url):
	A=url
	if A is _B:return''
	A=A.replace('.git','');return A
def install_extension_from_url(dirname,url,branch_name=_B):
	H='blob:none';I=branch_name;B=dirname;A=url;check_access()
	if isinstance(B,str):B=B.strip()
	if isinstance(A,str):A=A.strip()
	assert A,'No URL specified'
	if B is _B or B=='':*M,F=A.split('/');F=normalize_git_url(F);B=F
	C=os.path.join(extensions.extensions_dir,B);assert not os.path.exists(C),f"Extension directory already exists: {C}";K=normalize_git_url(A)
	if any(A for A in extensions.extensions if normalize_git_url(A.remote)==K):raise Exception(f"Extension with this URL is already installed: {A}")
	D=os.path.join(paths.data_path,'tmp',B)
	try:
		shutil.rmtree(D,_C)
		if not I:
			with git.Repo.clone_from(A,D,filter=[H])as E:
				E.remote().fetch()
				for G in E.submodules:G.update()
		else:
			with git.Repo.clone_from(A,D,filter=[H],branch=I)as E:
				E.remote().fetch()
				for G in E.submodules:G.update()
		try:os.rename(D,C)
		except OSError as J:
			if J.errno==errno.EXDEV:shutil.move(D,C)
			else:raise J
		import launch as L;L.run_extension_installer(C);extensions.list_extensions();return[extension_table(),html.escape(f"Installed into {C}. Use Installed tab to restart.")]
	finally:shutil.rmtree(D,_C)
def install_extension_from_index(url,hide_tags,sort_column,filter_text):A,B=install_extension_from_url(_B,url);C,D=refresh_available_extensions_from_data(hide_tags,sort_column,filter_text);return C,A,B,''
def refresh_available_extensions(url,hide_tags,sort_column):
	global available_extensions;import urllib.request
	with urllib.request.urlopen(url)as A:B=A.read()
	available_extensions=json.loads(B);C,D=refresh_available_extensions_from_data(hide_tags,sort_column);return url,C,gr.CheckboxGroup.update(choices=D),'',''
def refresh_available_extensions_for_tags(hide_tags,sort_column,filter_text):A,B=refresh_available_extensions_from_data(hide_tags,sort_column,filter_text);return A,''
def search_extensions(filter_text,hide_tags,sort_column):A,B=refresh_available_extensions_from_data(hide_tags,sort_column,filter_text);return A,''
sort_ordering=[(_C,lambda x:x.get(_L,'z')),(_A,lambda x:x.get(_L,'z')),(_A,lambda x:x.get(_G,'z')),(_C,lambda x:x.get(_G,'z')),(_A,lambda x:'z'),(_C,lambda x:x.get(_Q,'')),(_C,lambda x:x.get(_K,'')),(_C,lambda x:x.get(_M,0))]
def get_date(info,key):
	try:return datetime.strptime(info.get(key),'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d')
	except(ValueError,TypeError):return''
def refresh_available_extensions_from_data(hide_tags,sort_column,filter_text=''):
	I='tags';J=sort_column;C=filter_text;N=available_extensions[_E];O={normalize_git_url(A.remote):A.name for A in extensions.extensions};D=available_extensions.get(I,{});P=set(hide_tags);E=0;F=f'''<!-- {time.time()} -->
    <table id="available_extensions">
        <thead>
            <tr>
                <th>Extension</th>
                <th>Description</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
    ''';Q,R=sort_ordering[J if 0<=J<len(sort_ordering)else 0]
	for A in sorted(N,key=R,reverse=Q):
		K=A.get(_G,'noname');S=int(A.get(_M,0));T=A.get(_L,'unknown');U=get_date(A,_Q);V=get_date(A,_K);G=A.get('url',_B);L=A.get('description','');B=A.get(I,[])
		if G is _B:continue
		H=O.get(normalize_git_url(G),_B);B=B+[_H]if H else B
		if any(A for A in B if A in P):E+=1;continue
		if C and C.strip():
			if C.lower()not in html.escape(K).lower()and C.lower()not in html.escape(L).lower():E+=1;continue
		W=f'<button onclick="install_extension_from_index(this, \'{html.escape(G)}\')" {"disabled=disabled"if H else""} class="lg secondary gradio-button custom-button">{_R if not H else _S}</button>';X=', '.join([f"<span class='extension-tag' title='{D.get(A,'')}'>{A}</span>"for A in B]);F+=f'''
            <tr>
                <td><a href="{html.escape(G)}" target="_blank">{html.escape(K)}</a><br />{X}</td>
                <td>{html.escape(L)}<p class="info">
                <span class="date_added">Update: {html.escape(U)}  Added: {html.escape(T)}  Created: {html.escape(V)}</span><span class="star_count">stars: <b>{S}</b></a></p></td>
                <td>{W}</td>
            </tr>

        '''
		for M in[A for A in B if A not in D]:D[M]=M
	F+=_P
	if E>0:F+=f"<p>Extension hidden: {E}</p>"
	return F,list(D)
def preload_extensions_git_metadata():
	for A in extensions.extensions:A.read_info_from_repo()
def create_ui():
	L='newest first';M='localization';N='ads';O='Loading...';I='none';G='primary';import modules.ui;config_states.list_config_states();threading.Thread(target=preload_extensions_git_metadata).start()
	with gr.Blocks(analytics_enabled=_A)as J:
		with gr.Tabs(elem_id='tabs_extensions'):
			with gr.TabItem(_S,id=_H):
				with gr.Row(elem_id='extensions_installed_top'):W='Apply and restart UI'if restart.is_restartable()else'Apply and quit';X=gr.Button(value=W,variant=G);Y=gr.Button(value='Check for updates');Z=gr.Radio(label='Disable all extensions',choices=[I,'extra','all'],value=shared.opts.disable_all_extensions,elem_id='extensions_disable_all');P=gr.Text(elem_id='extensions_disabled_list',visible=_A,container=_A);a=gr.Text(elem_id='extensions_update_list',visible=_A,container=_A)
				Q=''
				if shared.cmd_opts.disable_all_extensions or shared.cmd_opts.disable_extra_extensions or shared.opts.disable_all_extensions!=I:
					if shared.cmd_opts.disable_all_extensions:K='"--disable-all-extensions" was used, remove it to load all extensions again'
					elif shared.opts.disable_all_extensions!=I:K='"Disable all extensions" was set, change it to "none" to load all extensions again'
					elif shared.cmd_opts.disable_extra_extensions:K='"--disable-extra-extensions" was used, remove it to load all extensions again'
					Q=f'<span style="color: var(--primary-400);">{K}</span>'
				with gr.Row():R=gr.HTML(Q)
				with gr.Row(elem_classes='progress-container'):H=gr.HTML(O,elem_id='extensions_installed_html')
				J.load(fn=extension_table,inputs=[],outputs=[H]);X.click(fn=apply_and_restart,_js='extensions_apply',inputs=[P,a,Z],outputs=[]);Y.click(fn=wrap_gradio_gpu_call(check_updates,extra_outputs=[gr.update()]),_js='extensions_check',inputs=[R,P],outputs=[H,R])
			with gr.TabItem('Available',id='available'):
				with gr.Row():b=gr.Button(value='Load from:',variant=G);c=os.environ.get('WEBUI_EXTENSIONS_INDEX','https://raw.githubusercontent.com/tenebo/standard-demo-we-extensions/master/index.json');S=gr.Text(value=c,label='Extension index URL',container=_A);d=gr.Text(elem_id='extension_to_install',visible=_A);e=gr.Button(elem_id='install_extension_button',visible=_A)
				with gr.Row():A=gr.CheckboxGroup(value=[N,M,_H],label='Hide extensions with tags',choices=['script',N,M,_H]);C=gr.Radio(value=L,label='Order',choices=[L,'oldest first','a-z','z-a','internal order','update time','create time',_M],type='index')
				with gr.Row():D=gr.Text(label='Search',container=_A)
				B=gr.HTML();F=gr.HTML();b.click(fn=modules.ui.wrap_gradio_call(refresh_available_extensions,extra_outputs=[gr.update(),gr.update(),gr.update(),gr.update()]),inputs=[S,A,C],outputs=[S,F,A,D,B]);e.click(fn=modules.ui.wrap_gradio_call(install_extension_from_index,extra_outputs=[gr.update(),gr.update()]),inputs=[d,A,C,D],outputs=[F,H,B]);D.change(fn=modules.ui.wrap_gradio_call(search_extensions,extra_outputs=[gr.update()]),inputs=[D,A,C],outputs=[F,B]);A.change(fn=modules.ui.wrap_gradio_call(refresh_available_extensions_for_tags,extra_outputs=[gr.update()]),inputs=[A,C,D],outputs=[F,B]);C.change(fn=modules.ui.wrap_gradio_call(refresh_available_extensions_for_tags,extra_outputs=[gr.update()]),inputs=[A,C,D],outputs=[F,B])
			with gr.TabItem('Install from URL',id='install_from_url'):T=gr.Text(label="URL for extension's git repository");f=gr.Text(label='Specific branch name',placeholder='Leave empty for default main branch');g=gr.Text(label='Local directory name',placeholder='Leave empty for auto');h=gr.Button(value=_R,variant=G);B=gr.HTML(elem_id='extension_install_result');h.click(fn=modules.ui.wrap_gradio_call(lambda*A:[gr.update(),*install_extension_from_url(*A)],extra_outputs=[gr.update(),gr.update()]),inputs=[g,T,f],outputs=[T,H,B])
			with gr.TabItem('Backup/Restore'):
				with gr.Row(elem_id='extensions_backup_top_row'):E=gr.Dropdown(label='Saved Configs',elem_id='extension_backup_saved_configs',value=_D,choices=[_D]+list(config_states.all_config_states.keys()));modules.ui.create_refresh_button(E,config_states.list_config_states,lambda:{'choices':[_D]+list(config_states.all_config_states.keys())},'refresh_config_states');i=gr.Radio(label='State to restore',choices=[_E,_F,_I],value=_E,elem_id='extension_backup_restore_type');j=gr.Button(value='Restore Selected Config',variant=G,elem_id='extension_backup_restore')
				with gr.Row(elem_id='extensions_backup_top_row2'):k=gr.Textbox('',placeholder='Config Name',show_label=_A);l=gr.Button(value='Save Current Config')
				U=gr.HTML('');V=gr.HTML(O);J.load(fn=update_config_states_table,inputs=[E],outputs=[V]);l.click(fn=save_config_state,inputs=[k],outputs=[E,U]);m=gr.Label(visible=_A);j.click(fn=restore_config_state,_js='config_state_confirm_restore',inputs=[m,E,i],outputs=[U]);E.change(fn=update_config_states_table,inputs=[E],outputs=[V])
	return J