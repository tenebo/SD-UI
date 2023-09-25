_B=False
_A=None
from functools import wraps
import html,time
from modules import shared,progress,errors,devices,fifo_lock
queue_lock=fifo_lock.FIFOLock()
def wrap_queued_call(func):
	def A(*A,**B):
		with queue_lock:C=func(*A,**B)
		return C
	return A
def wrap_gradio_gpu_call(func,extra_outputs=_A):
	@wraps(func)
	def A(*A,**D):
		if A and type(A[0])==str and A[0].startswith('task(')and A[0].endswith(')'):B=A[0];progress.add_task_to_queue(B)
		else:B=_A
		with queue_lock:
			shared.state.begin(job=B);progress.start_task(B)
			try:C=func(*A,**D);progress.record_results(B,C)
			finally:progress.finish_task(B)
			shared.state.end()
		return C
	return wrap_gradio_call(A,extra_outputs=extra_outputs,add_stats=True)
def wrap_gradio_call(func,extra_outputs=_A,add_stats=_B):
	G=add_stats
	@wraps(func)
	def A(*H,extra_outputs_array=extra_outputs,**I):
		D=extra_outputs_array;J=shared.opts.memmon_poll_rate>0 and not shared.mem_mon.disabled and G
		if J:shared.mem_mon.monitor()
		Q=time.perf_counter()
		try:A=list(func(*H,**I))
		except Exception as K:
			E=131072;R='Error completing request';B=f"Arguments: {H} {I}"[:E]
			if len(B)>E:B+=f" (Argument list truncated at {E}/{len(B)} characters)"
			errors.report(f"{R}\n{B}",exc_info=True);shared.state.job='';shared.state.job_count=0
			if D is _A:D=[_A,'']
			S=f"{type(K).__name__}: {K}";A=D+[f"<div class='error'>{html.escape(S)}</div>"]
		devices.torch_gc();shared.state.skipped=_B;shared.state.interrupted=_B;shared.state.job_count=0
		if not G:return tuple(A)
		L=time.perf_counter()-Q;M=int(L//60);T=L%60;F=f"{T:.1f} sec."
		if M>0:F=f"{M} min. "+F
		if J:C={A:-(B//-(1024*1024))for(A,B)in shared.mem_mon.stop().items()};U=C['active_peak'];V=C['reserved_peak'];N=C['system_peak'];O=C['total'];W=N/max(O,1)*100;X='Active: peak amount of video memory used during generation (excluding cached data)';Y='Reserved: total amout of video memory allocated by the Torch library ';Z='System: peak amout of video memory allocated by all running programs, out of total capacity';a=f"<abbr title='{X}'>A</abbr>: <span class='measurement'>{U/1024:.2f} GB</span>";b=f"<abbr title='{Y}'>R</abbr>: <span class='measurement'>{V/1024:.2f} GB</span>";c=f"<abbr title='{Z}'>Sys</abbr>: <span class='measurement'>{N/1024:.1f}/{O/1024:g} GB</span> ({W:.1f}%)";P=f"<p class='vram'>{a}, <wbr>{b}, <wbr>{c}</p>"
		else:P=''
		A[-1]+=f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{F}</span></p>{P}</div>";return tuple(A)
	return A