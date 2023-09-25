_B='min_free'
_A=None
import threading,time
from collections import defaultdict
import torch
class MemUsageMonitor(threading.Thread):
	run_flag=_A;device=_A;disabled=False;opts=_A;data=_A
	def __init__(A,name,device,opts):
		threading.Thread.__init__(A);A.name=name;A.device=device;A.opts=opts;A.daemon=True;A.run_flag=threading.Event();A.data=defaultdict(int)
		try:A.cuda_mem_get_info();torch.cuda.memory_stats(A.device)
		except Exception as B:print(f"Warning: caught exception '{B}', memory monitor disabled");A.disabled=True
	def cuda_mem_get_info(A):B=A.device.index if A.device.index is not _A else torch.cuda.current_device();return torch.cuda.mem_get_info(B)
	def run(A):
		if A.disabled:return
		while True:
			A.run_flag.wait();torch.cuda.reset_peak_memory_stats();A.data.clear()
			if A.opts.memmon_poll_rate<=0:A.run_flag.clear();continue
			A.data[_B]=A.cuda_mem_get_info()[0]
			while A.run_flag.is_set():B,C=A.cuda_mem_get_info();A.data[_B]=min(A.data[_B],B);time.sleep(1/A.opts.memmon_poll_rate)
	def dump_debug(B):
		print(B,'recorded data:')
		for(A,C)in B.read().items():print(A,-(C//-1024**2))
		print(B,'raw torch memory stats:');D=torch.cuda.memory_stats(B.device)
		for(A,C)in D.items():
			if'bytes'not in A:continue
			print('\t'if'peak'in A else'',A,-(C//-1024**2))
		print(torch.cuda.memory_summary())
	def monitor(A):A.run_flag.set()
	def read(A):
		if not A.disabled:D,C=A.cuda_mem_get_info();A.data['free']=D;A.data['total']=C;B=torch.cuda.memory_stats(A.device);A.data['active']=B['active.all.current'];A.data['active_peak']=B['active_bytes.all.peak'];A.data['reserved']=B['reserved_bytes.all.current'];A.data['reserved_peak']=B['reserved_bytes.all.peak'];A.data['system_peak']=C-A.data[_B]
		return A.data
	def stop(A):A.run_flag.clear();return A.read()