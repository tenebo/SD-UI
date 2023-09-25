import os,importlib.util
from modules import errors
def load_module(path):A=importlib.util.spec_from_file_location(os.path.basename(path),path);B=importlib.util.module_from_spec(A);A.loader.exec_module(B);return B
def preload_extensions(extensions_dir,parser,extension_list=None):
	C=extension_list;A=extensions_dir
	if not os.path.isdir(A):return
	E=C if C is not None else os.listdir(A)
	for F in sorted(E):
		B=os.path.join(A,F,'preload.py')
		if not os.path.isfile(B):continue
		try:
			D=load_module(B)
			if hasattr(D,'preload'):D.preload(parser)
		except Exception:errors.report(f"Error running preload() for {B}",exc_info=True)