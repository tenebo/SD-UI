import threading,collections
class FIFOLock:
	def __init__(A):A._lock=threading.Lock();A._inner_lock=threading.Lock();A._pending_threads=collections.deque()
	def acquire(A,blocking=True):
		C=False
		with A._inner_lock:
			D=A._lock.acquire(C)
			if D:return True
			elif not blocking:return C
			B=threading.Event();A._pending_threads.append(B)
		B.wait();return A._lock.acquire()
	def release(A):
		with A._inner_lock:
			if A._pending_threads:B=A._pending_threads.popleft();B.set()
			A._lock.release()
	__enter__=acquire
	def __exit__(A,t,v,tb):A.release()