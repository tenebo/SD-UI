_B='Live preview image ID'
_A=None
import base64,io,time,gradio as gr
from pydantic import BaseModel,Field
from modules.shared import opts
import modules.shared as shared
current_task=_A
pending_tasks={}
finished_tasks=[]
recorded_results=[]
recorded_results_limit=2
def start_task(id_task):A=id_task;global current_task;current_task=A;pending_tasks.pop(A,_A)
def finish_task(id_task):
	A=id_task;global current_task
	if current_task==A:current_task=_A
	finished_tasks.append(A)
	if len(finished_tasks)>16:finished_tasks.pop(0)
def record_results(id_task,res):
	recorded_results.append((id_task,res))
	if len(recorded_results)>recorded_results_limit:recorded_results.pop(0)
def add_task_to_queue(id_job):pending_tasks[id_job]=time.time()
class ProgressRequest(BaseModel):id_task=Field(default=_A,title='Task ID',description='id of the task to get progress for');id_live_preview=Field(default=-1,title=_B,description='id of last received last preview image');live_preview=Field(default=True,title='Include live preview',description='boolean flag indicating whether to include the live preview image')
class ProgressResponse(BaseModel):active=Field(title='Whether the task is being worked on right now');queued=Field(title='Whether the task is in queue');completed=Field(title='Whether the task has already finished');progress=Field(default=_A,title='Progress',description='The progress with a range of 0 to 1');eta=Field(default=_A,title='ETA in secs');live_preview=Field(default=_A,title='Live preview image',description='Current live preview; a data: uri');id_live_preview=Field(default=_A,title=_B,description='Send this together with next request to prevent receiving same image');textinfo=Field(default=_A,title='Info text',description='Info text used by WebUI.')
def setup_progress_api(app):return app.add_api_route('/internal/progress',progressapi,methods=['POST'],response_model=ProgressResponse)
def progressapi(req):
	Q='optimize';A=req;D=A.id_task==current_task;E=A.id_task in pending_tasks;H=A.id_task in finished_tasks
	if not D:
		I='Waiting...'
		if E:J=sorted(pending_tasks.keys(),key=lambda x:pending_tasks[x]);R=J.index(A.id_task);I='In queue: {}/{}'.format(R+1,len(J))
		return ProgressResponse(active=D,queued=E,completed=H,id_live_preview=-1,textinfo=I)
	B=0;C,S=shared.state.job_count,shared.state.job_no;K,T=shared.state.sampling_steps,shared.state.sampling_step
	if C>0:B+=S/C
	if K>0 and C>0:B+=1/C*T/K
	B=min(B,1);L=time.time()-shared.state.time_start;M=L/B if B>0 else _A;U=M-L if M is not _A else _A;N=_A;O=A.id_live_preview
	if opts.live_previews_enable and A.live_preview:
		shared.state.set_current_image()
		if shared.state.id_live_preview!=A.id_live_preview:
			F=shared.state.current_image
			if F is not _A:
				P=io.BytesIO()
				if opts.live_previews_image_format=='png':
					if max(*F.size)<=256:G={Q:True}
					else:G={Q:False,'compress_level':1}
				else:G={}
				F.save(P,format=opts.live_previews_image_format,**G);V=base64.b64encode(P.getvalue()).decode('ascii');N=f"data:image/{opts.live_previews_image_format};base64,{V}";O=shared.state.id_live_preview
	return ProgressResponse(active=D,queued=E,completed=H,progress=B,eta=U,live_preview=N,id_live_preview=O,textinfo=shared.state.textinfo)
def restore_progress(id_task):
	A=id_task
	while A==current_task or A in pending_tasks:time.sleep(.1)
	B=next(iter([B[1]for B in recorded_results if A==B[0]]),_A)
	if B is not _A:return B
	return gr.update(),gr.update(),gr.update(),f"Couldn't restore progress for {A}: results either have been discarded or never were obtained"