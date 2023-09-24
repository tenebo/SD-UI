_A=True
import os
from PIL import Image
from modules import shared,images,devices,scripts,scripts_postprocessing,ui_common,generation_parameters_copypaste
from modules.shared import opts
def run_postprocessing(extras_mode,image,image_folder,input_dir,output_dir,show_extras_results,*O,save_output=_A):
	I='extras';J=output_dir;F=extras_mode;C='';B=None;devices.torch_gc();shared.state.begin(job=I);K=[]
	def P(extras_mode,image,image_folder,input_dir):
		E=input_dir;F=extras_mode;A=image
		if F==1:
			for D in image_folder:
				if isinstance(D,Image.Image):A=D;G=C
				else:A=Image.open(os.path.abspath(D.name));G=os.path.splitext(D.orig_name)[0]
				yield(A,G)
		elif F==2:
			assert not shared.cmd_opts.hide_ui_dir_config,'--hide-ui-dir-config option must be disabled';assert E,'input directory not selected';I=shared.listfiles(E)
			for H in I:
				try:A=Image.open(H)
				except Exception:continue
				yield(A,H)
		else:assert A,'image not selected';yield(A,B)
	if F==2 and J!=C:L=J
	else:L=opts.outdir_samples or opts.outdir_extras_samples
	D=C
	for(E,G)in P(F,image,image_folder,input_dir):
		E:0;shared.state.textinfo=G;M,H=images.read_info_from_image(E)
		if M:H['parameters']=M
		A=scripts_postprocessing.PostprocessedImage(E.convert('RGB'));scripts.scripts_postproc.run(A,O)
		if opts.use_original_name_batch and G is not B:N=os.path.splitext(os.path.basename(G))[0]
		else:N=C
		D=', '.join([A if A==C else f"{A}: {generation_parameters_copypaste.quote(C)}"for(A,C)in A.info.items()if C is not B])
		if opts.enable_pnginfo:A.image.info=H;A.image.info['postprocessing']=D
		if save_output:images.save_image(A.image,path=L,basename=N,seed=B,prompt=B,extension=opts.samples_format,info=D,short_filename=_A,no_prompt=_A,grid=False,pnginfo_section_name=I,existing_info=H,forced_filename=B)
		if F!=2 or show_extras_results:K.append(A.image)
		E.close()
	devices.torch_gc();return K,ui_common.plaintext_to_html(D),C
def run_extras(extras_mode,resize_mode,image,image_folder,input_dir,output_dir,show_extras_results,gfpgan_visibility,codeformer_visibility,codeformer_weight,upscaling_resize,upscaling_resize_w,upscaling_resize_h,upscaling_crop,extras_upscaler_1,extras_upscaler_2,extras_upscaler_2_visibility,upscale_first,save_output=_A):'old handler for API';A=scripts.scripts_postproc.create_args_for_run({'Upscale':{'upscale_mode':resize_mode,'upscale_by':upscaling_resize,'upscale_to_width':upscaling_resize_w,'upscale_to_height':upscaling_resize_h,'upscale_crop':upscaling_crop,'upscaler_1_name':extras_upscaler_1,'upscaler_2_name':extras_upscaler_2,'upscaler_2_visibility':extras_upscaler_2_visibility},'GFPGAN':{'gfpgan_visibility':gfpgan_visibility},'CodeFormer':{'codeformer_visibility':codeformer_visibility,'codeformer_weight':codeformer_weight}});return run_postprocessing(extras_mode,image,image_folder,input_dir,output_dir,show_extras_results,*A,save_output=save_output)