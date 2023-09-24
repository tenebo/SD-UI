_D='temp_dirs'
_C='temp_file_sets'
_B=None
_A=False
import os,tempfile
from collections import namedtuple
from pathlib import Path
import gradio.components
from PIL import PngImagePlugin
from modules import shared
Savedfile=namedtuple('Savedfile',['name'])
def register_tmp_file(gradio,filename):
	B=filename;A=gradio
	if hasattr(A,_C):A.temp_file_sets[0]=A.temp_file_sets[0]|{os.path.abspath(B)}
	if hasattr(A,_D):A.temp_dirs=A.temp_dirs|{os.path.abspath(os.path.dirname(B))}
def check_tmp_file(gradio,filename):
	B=filename;A=gradio
	if hasattr(A,_C):return any(B in A for A in A.temp_file_sets)
	if hasattr(A,_D):return any(Path(A).resolve()in Path(B).resolve().parents for A in A.temp_dirs)
	return _A
def save_pil_to_file(self,pil_image,dir=_B,format='png'):
	B=pil_image;A=getattr(B,'already_saved_as',_B)
	if A and os.path.isfile(A):
		register_tmp_file(shared.demo,A);C=A
		if not shared.opts.save_images_add_number:C+=f"?{os.path.getmtime(A)}"
		return C
	if shared.opts.temp_dir!='':dir=shared.opts.temp_dir
	else:os.makedirs(dir,exist_ok=True)
	D=_A;E=PngImagePlugin.PngInfo()
	for(F,G)in B.info.items():
		if isinstance(F,str)and isinstance(G,str):E.add_text(F,G);D=True
	H=tempfile.NamedTemporaryFile(delete=_A,suffix='.png',dir=dir);B.save(H,pnginfo=E if D else _B);return H.name
def install_ui_tempdir_override():'override save to file function so that it also writes PNG info';gradio.components.IOComponent.pil_to_temp_file=save_pil_to_file
def on_tmpdir_changed():
	if shared.opts.temp_dir==''or shared.demo is _B:return
	os.makedirs(shared.opts.temp_dir,exist_ok=True);register_tmp_file(shared.demo,os.path.join(shared.opts.temp_dir,'x'))
def cleanup_tmpdr():
	A=shared.opts.temp_dir
	if A==''or not os.path.isdir(A):return
	for(C,D,E)in os.walk(A,topdown=_A):
		for B in E:
			D,F=os.path.splitext(B)
			if F!='.png':continue
			G=os.path.join(C,B);os.remove(G)