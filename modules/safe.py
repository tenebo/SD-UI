_A=None
import pickle,collections,torch,numpy,_codecs,zipfile,re
from modules import errors
TypedStorage=torch.storage.TypedStorage if hasattr(torch.storage,'TypedStorage')else torch.storage._TypedStorage
def encode(*A):B=_codecs.encode(*A);return B
class RestrictedUnpickler(pickle.Unpickler):
	extra_handler=_A
	def persistent_load(A,saved_id):
		assert saved_id[0]=='storage'
		try:return TypedStorage(_internal=True)
		except TypeError:return TypedStorage()
	def find_class(C,module,name):
		B=module;A=name
		if C.extra_handler is not _A:
			D=C.extra_handler(B,A)
			if D is not _A:return D
		if B=='collections'and A=='OrderedDict':return getattr(collections,A)
		if B=='torch._utils'and A in['_rebuild_tensor_v2','_rebuild_parameter','_rebuild_device_tensor_from_numpy']:return getattr(torch._utils,A)
		if B=='torch'and A in['FloatStorage','HalfStorage','IntStorage','LongStorage','DoubleStorage','ByteStorage','float32','BFloat16Storage']:return getattr(torch,A)
		if B=='torch.nn.modules.container'and A in['ParameterDict']:return getattr(torch.nn.modules.container,A)
		if B=='numpy.core.multiarray'and A in['scalar','_reconstruct']:return getattr(numpy.core.multiarray,A)
		if B=='numpy'and A in['dtype','ndarray']:return getattr(numpy,A)
		if B=='_codecs'and A=='encode':return encode
		if B=='pytorch_lightning.callbacks'and A=='model_checkpoint':import pytorch_lightning.callbacks;return pytorch_lightning.callbacks.model_checkpoint
		if B=='pytorch_lightning.callbacks.model_checkpoint'and A=='ModelCheckpoint':import pytorch_lightning.callbacks.model_checkpoint;return pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
		if B=='__builtin__'and A=='set':return set
		raise Exception(f"global '{B}/{A}' is forbidden")
allowed_zip_names_re=re.compile('^([^/]+)/((data/\\d+)|version|(data\\.pkl))$')
data_pkl_re=re.compile('^([^/]+)/data\\.pkl$')
def check_zip_filenames(filename,names):
	for A in names:
		if allowed_zip_names_re.match(A):continue
		raise Exception(f"bad file inside {filename}: {A}")
def check_pt(filename,extra_handler):
	F=extra_handler;A=filename
	try:
		with zipfile.ZipFile(A)as C:
			check_zip_filenames(A,C.namelist());D=[A for A in C.namelist()if data_pkl_re.match(A)]
			if len(D)==0:raise Exception(f"data.pkl not found in {A}")
			if len(D)>1:raise Exception(f"Multiple data.pkl found in {A}")
			with C.open(D[0])as E:B=RestrictedUnpickler(E);B.extra_handler=F;B.load()
	except zipfile.BadZipfile:
		with open(A,'rb')as E:
			B=RestrictedUnpickler(E);B.extra_handler=F
			for G in range(5):B.load()
def load(filename,*A,**B):return load_with_extra(filename,*A,extra_handler=global_extra_handler,**B)
def load_with_extra(filename,extra_handler=_A,*B,**C):
	"\n    this function is intended to be used by extensions that want to load models with\n    some extra classes in them that the usual unpickler would find suspicious.\n\n    Use the extra_handler argument to specify a function that takes module and field name as text,\n    and returns that field's value:\n\n    ```python\n    def extra(module, name):\n        if module == 'collections' and name == 'OrderedDict':\n            return collections.OrderedDict\n\n        return None\n\n    safe.load_with_extra('model.pt', extra_handler=extra)\n    ```\n\n    The alternative to this is just to use safe.unsafe_torch_load('model.pt'), which as the name implies is\n    definitely unsafe.\n    ";A=filename;from modules import shared as D
	try:
		if not D.cmd_opts.disable_safe_unpickle:check_pt(A,extra_handler)
	except pickle.UnpicklingError:errors.report(f"Error verifying pickled file from {A}\n-----> !!!! The file is most likely corrupted !!!! <-----\nYou can skip this check with --disable-safe-unpickle commandline argument, but that is not going to help you.\n\n",exc_info=True);return
	except Exception:errors.report(f"Error verifying pickled file from {A}\nThe file may be malicious, so the program is not going to read it.\nYou can skip this check with --disable-safe-unpickle commandline argument.\n\n",exc_info=True);return
	return unsafe_torch_load(A,*B,**C)
class Extra:
	"\n    A class for temporarily setting the global handler for when you can't explicitly call load_with_extra\n    (because it's not your code making the torch.load call). The intended use is like this:\n\n```\nimport torch\nfrom modules import safe\n\ndef handler(module, name):\n    if module == 'torch' and name in ['float64', 'float16']:\n        return getattr(torch, name)\n\n    return None\n\nwith safe.Extra(handler):\n    x = torch.load('model.pt')\n```\n    "
	def __init__(A,handler):A.handler=handler
	def __enter__(A):global global_extra_handler;assert global_extra_handler is _A,'already inside an Extra() block';global_extra_handler=A.handler
	def __exit__(A,exc_type,exc_val,exc_tb):global global_extra_handler;global_extra_handler=_A
unsafe_torch_load=torch.load
torch.load=load
global_extra_handler=_A