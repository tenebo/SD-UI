from __future__ import annotations
_B=True
_A=None
import os,shutil,importlib
from urllib.parse import urlparse
from modules import shared
from modules.upscaler import Upscaler,UpscalerLanczos,UpscalerNearest,UpscalerNone
from modules.paths import script_path,models_path
def load_file_from_url(url,*,model_dir,progress=_B,file_name=_A):
	'Download a file from `url` into `model_dir`, using the file present if possible.\n\n    Returns the path to the downloaded file.\n    ';os.makedirs(model_dir,exist_ok=_B)
	if not file_name:parts=urlparse(url);file_name=os.path.basename(parts.path)
	cached_file=os.path.abspath(os.path.join(model_dir,file_name))
	if not os.path.exists(cached_file):print(f'Downloading: "{url}" to {cached_file}\n');from torch.hub import download_url_to_file;download_url_to_file(url,cached_file,progress=progress)
	return cached_file
def load_models(model_path,model_url=_A,command_path=_A,ext_filter=_A,download_name=_A,ext_blacklist=_A):
	'\n    A one-and done loader to try finding the desired models in specified directories.\n\n    @param download_name: Specify to download from model_url immediately.\n    @param model_url: If no other models are found, this will be downloaded on upscale.\n    @param model_path: The location to store/find models in.\n    @param command_path: A command-line argument to search for models in first.\n    @param ext_filter: An optional list of filename extensions to filter by\n    @return: A list of paths containing the desired model(s)\n    ';output=[]
	try:
		places=[]
		if command_path is not _A and command_path!=model_path:
			pretrained_path=os.path.join(command_path,'experiments/pretrained_models')
			if os.path.exists(pretrained_path):print(f"Appending path: {pretrained_path}");places.append(pretrained_path)
			elif os.path.exists(command_path):places.append(command_path)
		places.append(model_path)
		for place in places:
			for full_path in shared.walk_files(place,allowed_extensions=ext_filter):
				if os.path.islink(full_path)and not os.path.exists(full_path):print(f"Skipping broken symlink: {full_path}");continue
				if ext_blacklist is not _A and any(full_path.endswith(x)for x in ext_blacklist):continue
				if full_path not in output:output.append(full_path)
		if model_url is not _A and len(output)==0:
			if download_name is not _A:output.append(load_file_from_url(model_url,model_dir=places[0],file_name=download_name))
			else:output.append(model_url)
	except Exception:pass
	return output
def friendly_name(file):
	if file.startswith('http'):file=urlparse(file).path
	file=os.path.basename(file);model_name,extension=os.path.splitext(file);return model_name
def cleanup_models():B='SwinIR';A='ESRGAN';root_path=script_path;src_path=models_path;dest_path=os.path.join(models_path,'Standard-demo');move_files(src_path,dest_path,'.ckpt');move_files(src_path,dest_path,'.safetensors');src_path=os.path.join(root_path,A);dest_path=os.path.join(models_path,A);move_files(src_path,dest_path);src_path=os.path.join(models_path,'BSRGAN');dest_path=os.path.join(models_path,A);move_files(src_path,dest_path,'.pth');src_path=os.path.join(root_path,'gfpgan');dest_path=os.path.join(models_path,'GFPGAN');move_files(src_path,dest_path);src_path=os.path.join(root_path,B);dest_path=os.path.join(models_path,B);move_files(src_path,dest_path);src_path=os.path.join(root_path,'repositories/latent-diffusion/experiments/pretrained_models/');dest_path=os.path.join(models_path,'LDSR');move_files(src_path,dest_path)
def move_files(src_path,dest_path,ext_filter=_A):
	try:
		os.makedirs(dest_path,exist_ok=_B)
		if os.path.exists(src_path):
			for file in os.listdir(src_path):
				fullpath=os.path.join(src_path,file)
				if os.path.isfile(fullpath):
					if ext_filter is not _A:
						if ext_filter not in file:continue
					print(f"Moving {file} from {src_path} to {dest_path}.")
					try:shutil.move(fullpath,dest_path)
					except Exception:pass
			if len(os.listdir(src_path))==0:print(f"Removing empty folder: {src_path}");shutil.rmtree(src_path,_B)
	except Exception:pass
def load_upscalers():
	A='_model.py';modules_dir=os.path.join(shared.script_path,'modules')
	for file in os.listdir(modules_dir):
		if A in file:
			model_name=file.replace(A,'');full_model=f"modules.{model_name}_model"
			try:importlib.import_module(full_model)
			except Exception:pass
	datas=[];commandline_options=vars(shared.cmd_opts);used_classes={}
	for cls in reversed(Upscaler.__subclasses__()):
		classname=str(cls)
		if classname not in used_classes:used_classes[classname]=cls
	for cls in reversed(used_classes.values()):name=cls.__name__;cmd_name=f"{name.lower().replace('upscaler','')}_models_path";commandline_model_path=commandline_options.get(cmd_name,_A);scaler=cls(commandline_model_path);scaler.user_path=commandline_model_path;scaler.model_download_path=commandline_model_path or scaler.model_path;datas+=scaler.scalers
	shared.sd_upscalers=sorted(datas,key=lambda x:x.name.lower()if not isinstance(x.scaler,(UpscalerNone,UpscalerLanczos,UpscalerNearest))else'')