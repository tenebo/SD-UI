_A=None
import tqdm
from modules import shared
class TotalTQDM:
	def __init__(A):A._tqdm=_A
	def reset(A):A._tqdm=tqdm.tqdm(desc='Total progress',total=shared.state.job_count*shared.state.sampling_steps,position=1,file=shared.progress_print_out)
	def update(A):
		if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:return
		if A._tqdm is _A:A.reset()
		A._tqdm.update()
	def updateTotal(A,new_total):
		if not shared.opts.multiple_tqdm or shared.cmd_opts.disable_console_progressbars:return
		if A._tqdm is _A:A.reset()
		A._tqdm.total=new_total
	def clear(A):
		if A._tqdm is not _A:A._tqdm.refresh();A._tqdm.close();A._tqdm=_A