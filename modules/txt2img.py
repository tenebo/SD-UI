from contextlib import closing
import modules.scripts
from modules import processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.shared import opts,cmd_opts
import modules.shared as shared
from modules.ui import plaintext_to_html
import gradio as gr
def txt2img(id_task,prompt,negative_prompt,prompt_styles,steps,sampler_name,n_iter,batch_size,cfg_scale,height,width,enable_hr,denoising_strength,hr_scale,hr_upscaler,hr_second_pass_steps,hr_resize_x,hr_resize_y,hr_checkpoint_name,hr_sampler_name,hr_prompt,hr_negative_prompt,override_settings_texts,request,*H):
	G=hr_sampler_name;F=hr_checkpoint_name;E=enable_hr;D=prompt;C=None;J=create_override_settings_dict(override_settings_texts);B=processing.StandardDemoProcessingTxt2Img(sd_model=shared.sd_model,outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,prompt=D,styles=prompt_styles,negative_prompt=negative_prompt,sampler_name=sampler_name,batch_size=batch_size,n_iter=n_iter,steps=steps,cfg_scale=cfg_scale,width=width,height=height,enable_hr=E,denoising_strength=denoising_strength if E else C,hr_scale=hr_scale,hr_upscaler=hr_upscaler,hr_second_pass_steps=hr_second_pass_steps,hr_resize_x=hr_resize_x,hr_resize_y=hr_resize_y,hr_checkpoint_name=C if F=='Use same checkpoint'else F,hr_sampler_name=C if G=='Use same sampler'else G,hr_prompt=hr_prompt,hr_negative_prompt=hr_negative_prompt,override_settings=J);B.scripts=modules.scripts.scripts_txt2img;B.script_args=H;B.user=request.username
	if cmd_opts.enable_console_prompts:print(f"\ntxt2img: {D}",file=shared.progress_print_out)
	with closing(B):
		A=modules.scripts.scripts_txt2img.run(B,*H)
		if A is C:A=processing.process_images(B)
	shared.total_tqdm.clear();I=A.js()
	if opts.samples_log_stdout:print(I)
	if opts.do_not_show_images:A.images=[]
	return A.images,I,plaintext_to_html(A.info),plaintext_to_html(A.comments,classname='comments')