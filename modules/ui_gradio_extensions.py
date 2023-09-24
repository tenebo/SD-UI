import os,gradio as gr
from modules import localization,shared,scripts
from modules.paths import script_path,data_path
def webpath(fn):
	if fn.startswith(script_path):A=os.path.relpath(fn,script_path).replace('\\','/')
	else:A=os.path.abspath(fn)
	return f"file={A}?{os.path.getmtime(fn)}"
def javascript_html():
	C='javascript';A=f'<script type="text/javascript">{localization.localization_js(shared.opts.localization)}</script>\n';D=os.path.join(script_path,'script.js');A+=f'<script type="text/javascript" src="{webpath(D)}"></script>\n'
	for B in scripts.list_scripts(C,'.js'):A+=f'<script type="text/javascript" src="{webpath(B.path)}"></script>\n'
	for B in scripts.list_scripts(C,'.mjs'):A+=f'<script type="module" src="{webpath(B.path)}"></script>\n'
	if shared.cmd_opts.theme:A+=f'<script type="text/javascript">set_theme("{shared.cmd_opts.theme}");</script>\n'
	return A
def css_html():
	B='user.css';A=''
	def C(fn):return f'<link rel="stylesheet" property="stylesheet" href="{webpath(fn)}">'
	for D in scripts.list_files_with_name('style.css'):
		if not os.path.isfile(D):continue
		A+=C(D)
	if os.path.exists(os.path.join(data_path,B)):A+=C(os.path.join(data_path,B))
	return A
def reload_javascript():
	C=javascript_html();D=css_html()
	def A(*E,**F):B='utf8';A=shared.GradioTemplateResponseOriginal(*E,**F);A.body=A.body.replace(b'</head>',f"{C}</head>".encode(B));A.body=A.body.replace(b'</body>',f"{D}</body>".encode(B));A.init_headers();return A
	gr.routes.templates.TemplateResponse=A
if not hasattr(shared,'GradioTemplateResponseOriginal'):shared.GradioTemplateResponseOriginal=gr.routes.templates.TemplateResponse