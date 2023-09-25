_C='sha256'
_B='hashes'
_A='hashes-addnet'
import hashlib,os.path
from modules import shared
import modules.cache
dump_cache=modules.cache.dump_cache
cache=modules.cache.cache
def calculate_sha256(filename):
	A=hashlib.sha256();B=1024*1024
	with open(filename,'rb')as C:
		for D in iter(lambda:C.read(B),b''):A.update(D)
	return A.hexdigest()
def sha256_from_cache(filename,title,use_addnet_hash=False):
	A=title;B=cache(_A)if use_addnet_hash else cache(_B);D=os.path.getmtime(filename)
	if A not in B:return
	C=B[A].get(_C,None);E=B[A].get('mtime',0)
	if D>E or C is None:return
	return C
def sha256(filename,title,use_addnet_hash=False):
	D=title;C=use_addnet_hash;B=filename;E=cache(_A)if C else cache(_B);A=sha256_from_cache(B,D,C)
	if A is not None:return A
	if shared.cmd_opts.no_hashing:return
	print(f"Calculating sha256 for {B}: ",end='')
	if C:
		with open(B,'rb')as F:A=addnet_hash_safetensors(F)
	else:A=calculate_sha256(B)
	print(f"{A}");E[D]={'mtime':os.path.getmtime(B),_C:A};dump_cache();return A
def addnet_hash_safetensors(b):
	'kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py';A=hashlib.sha256();B=1024*1024;b.seek(0);C=b.read(8);D=int.from_bytes(C,'little');E=D+8;b.seek(E)
	for F in iter(lambda:b.read(B),b''):A.update(F)
	return A.hexdigest()