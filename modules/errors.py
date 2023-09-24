import sys,textwrap,traceback
exception_records=[]
def record_exception():
	D,A,B=sys.exc_info()
	if A is None:return
	if exception_records and exception_records[-1]==A:return
	from modules import sysinfo as C;exception_records.append(C.format_exception(A,B))
	if len(exception_records)>5:exception_records.pop(0)
def report(message,*,exc_info=False):
	'\n    Print an error message to stderr, with optional traceback.\n    ';record_exception()
	for A in message.splitlines():print('***',A,file=sys.stderr)
	if exc_info:print(textwrap.indent(traceback.format_exc(),'    '),file=sys.stderr);print('---',file=sys.stderr)
def print_error_explanation(message):
	record_exception();A=message.strip().split('\n');B=max([len(A)for A in A]);print('='*B,file=sys.stderr)
	for C in A:print(C,file=sys.stderr)
	print('='*B,file=sys.stderr)
def display(e,task,*,full_traceback=False):
	record_exception();print(f"{task or'error'}: {type(e).__name__}",file=sys.stderr);A=traceback.TracebackException.from_exception(e)
	if full_traceback:A.stack=traceback.StackSummary(traceback.extract_stack()[:-2]+A.stack)
	print(*A.format(),sep='',file=sys.stderr);B=str(e)
	if'copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])'in B:print_error_explanation('\nThe most likely cause of this is you are trying to load Standard Demo 2.0 model without specifying its config file.\nSee https://github.com/tenebo/standard-demo-we/wiki/Features#standard-demo-20 for how to solve this.\n        ')
already_displayed={}
def display_once(e,task):
	A=task;record_exception()
	if A in already_displayed:return
	display(e,A);already_displayed[A]=1
def run(code,task):
	try:code()
	except Exception as A:display(task,A)
def check_versions():
	from packaging import version as A;from modules import shared as H;import torch as B,gradio as C;D='2.0.0';E='0.0.20';F='3.41.2'
	if A.parse(B.__version__)<A.parse(D):print_error_explanation(f"""
You are running torch {B.__version__}.
The program is tested to work with torch {D}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())
	if H.xformers_available:
		import xformers as G
		if A.parse(G.__version__)<A.parse(E):print_error_explanation(f"""
You are running xformers {G.__version__}.
The program is tested to work with xformers {E}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())
	if C.__version__!=F:print_error_explanation(f"""
You are running gradio {C.__version__}.
The program is designed to work with gradio {F}.
Using a different version of gradio is extremely likely to break the program.

Reasons why you have the mismatched gradio version can be:
  - you use --skip-install flag.
  - you use webui.py to start the program instead of launch.py.
  - an extension installs the incompatible gradio version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())