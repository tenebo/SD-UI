_C='restart'
_B=False
_A=None
import datetime,logging,threading,time
from modules import errors,shared,devices
from typing import Optional
log=logging.getLogger(__name__)
class State:
	skipped=_B;interrupted=_B;job='';job_no=0;job_count=0;processing_has_refined_job_count=_B;job_timestamp='0';sampling_step=0;sampling_steps=0;current_latent=_A;current_image=_A;current_image_sampling_step=0;id_live_preview=0;textinfo=_A;time_start=_A;server_start=_A;_server_command_signal=threading.Event();_server_command=_A
	def __init__(A):A.server_start=time.time()
	@property
	def need_restart(self):return self.server_command==_C
	@need_restart.setter
	def need_restart(self,value):
		if value:self.server_command=_C
	@property
	def server_command(self):return self._server_command
	@server_command.setter
	def server_command(self,value):"\n        Set the server command to `value` and signal that it's been set.\n        ";self._server_command=value;self._server_command_signal.set()
	def wait_for_server_command(A,timeout=_A):
		'\n        Wait for server command to get set; return and clear the value and signal.\n        '
		if A._server_command_signal.wait(timeout):A._server_command_signal.clear();B=A._server_command;A._server_command=_A;return B
	def request_restart(A):A.interrupt();A.server_command=_C;log.info('Received restart request')
	def skip(A):A.skipped=True;log.info('Received skip request')
	def interrupt(A):A.interrupted=True;log.info('Received interrupt request')
	def nextjob(A):
		if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps==-1:A.do_set_current_image()
		A.job_no+=1;A.sampling_step=0;A.current_image_sampling_step=0
	def dict(A):B={'skipped':A.skipped,'interrupted':A.interrupted,'job':A.job,'job_count':A.job_count,'job_timestamp':A.job_timestamp,'job_no':A.job_no,'sampling_step':A.sampling_step,'sampling_steps':A.sampling_steps};return B
	def begin(A,job='(unknown)'):A.sampling_step=0;A.job_count=-1;A.processing_has_refined_job_count=_B;A.job_no=0;A.job_timestamp=datetime.datetime.now().strftime('%Y%m%d%H%M%S');A.current_latent=_A;A.current_image=_A;A.current_image_sampling_step=0;A.id_live_preview=0;A.skipped=_B;A.interrupted=_B;A.textinfo=_A;A.time_start=time.time();A.job=job;devices.torch_gc();log.info('Starting job %s',job)
	def end(A):B=time.time()-A.time_start;log.info('Ending job %s (%.2f seconds)',A.job,B);A.job='';A.job_count=0;devices.torch_gc()
	def set_current_image(A):
		'if enough sampling steps have been made after the last call to this, sets self.current_image from self.current_latent, and modifies self.id_live_preview accordingly'
		if not shared.parallel_processing_allowed:return
		if A.sampling_step-A.current_image_sampling_step>=shared.opts.show_progress_every_n_steps and shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps!=-1:A.do_set_current_image()
	def do_set_current_image(A):
		if A.current_latent is _A:return
		import modules.sd_samplers
		try:
			if shared.opts.show_progress_grid:A.assign_current_image(modules.sd_samplers.samples_to_image_grid(A.current_latent))
			else:A.assign_current_image(modules.sd_samplers.sample_to_image(A.current_latent))
			A.current_image_sampling_step=A.sampling_step
		except Exception:errors.record_exception()
	def assign_current_image(A,image):A.current_image=image;A.id_live_preview+=1