import importlib
class CondFunc:
	def __new__(E,orig_func,sub_func,cond_func):
		B=orig_func;D=super(CondFunc,E).__new__(E)
		if isinstance(B,str):
			A=B.split('.')
			for F in range(len(A)-1,-1,-1):
				try:C=importlib.import_module('.'.join(A[:F]));break
				except ImportError:pass
			for G in A[F:-1]:C=getattr(C,G)
			B=getattr(C,A[-1]);setattr(C,A[-1],lambda*A,**B:D(*A,**B))
		D.__init__(B,sub_func,cond_func);return lambda*A,**B:D(*A,**B)
	def __init__(A,orig_func,sub_func,cond_func):A.__orig_func=orig_func;A.__sub_func=sub_func;A.__cond_func=cond_func
	def __call__(A,*B,**C):
		if not A.__cond_func or A.__cond_func(A.__orig_func,*B,**C):return A.__sub_func(A.__orig_func,*B,**C)
		else:return A.__orig_func(*B,**C)