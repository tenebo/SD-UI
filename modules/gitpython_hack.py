from __future__ import annotations
_A='cat-file'
import io,subprocess,git
class Git(git.Git):
	'\n    Git subclassed to never use persistent processes.\n    '
	def _get_persistent_cmd(C,attr_name,cmd_name,*A,**B):raise NotImplementedError(f"Refusing to use persistent process: {attr_name} ({cmd_name} {A} {B})")
	def get_object_header(A,ref):B=subprocess.check_output([A.GIT_PYTHON_GIT_EXECUTABLE,_A,'--batch-check'],input=A._prepare_ref(ref),cwd=A._working_dir,timeout=2);return A._parse_object_header(B)
	def stream_object_data(A,ref):D=subprocess.check_output([A.GIT_PYTHON_GIT_EXECUTABLE,_A,'--batch'],input=A._prepare_ref(ref),cwd=A._working_dir,timeout=30);B=io.BytesIO(D);E,F,C=A._parse_object_header(B.readline());return E,F,C,A.CatFileContentStream(C,B)
class Repo(git.Repo):GitCommandWrapperType=Git