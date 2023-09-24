import os,re,torch,numpy as np
from modules import modelloader,paths,deepbooru_model,devices,images,shared
re_special=re.compile('([\\\\()])')
class DeepDanbooru:
	def __init__(A):A.model=None
	def load(A):
		if A.model is not None:return
		B=modelloader.load_models(model_path=os.path.join(paths.models_path,'torch_deepdanbooru'),model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',ext_filter=['.pt'],download_name='model-resnet_custom_v3.pt');A.model=deepbooru_model.DeepDanbooruModel();A.model.load_state_dict(torch.load(B[0],map_location='cpu'));A.model.eval();A.model.to(devices.cpu,devices.dtype)
	def start(A):A.load();A.model.to(devices.device)
	def stop(A):
		if not shared.opts.interrogate_keep_models_in_memory:A.model.to(devices.cpu);devices.torch_gc()
	def tag(A,pil_image):A.start();B=A.tag_multi(pil_image);A.stop();return B
	def tag_multi(E,pil_image,force_disable_ranks=False):
		H=shared.opts.interrogate_deepbooru_score_threshold;I=shared.opts.deepbooru_use_spaces;J=shared.opts.deepbooru_escape;K=shared.opts.deepbooru_sort_alpha;L=shared.opts.interrogate_return_ranks and not force_disable_ranks;M=images.resize_image(2,pil_image.convert('RGB'),512,512);N=np.expand_dims(np.array(M,dtype=np.float32),0)/255
		with torch.no_grad(),devices.autocast():O=torch.from_numpy(N).to(devices.device);P=E.model(O)[0].detach().cpu().numpy()
		C={}
		for(B,D)in zip(E.model.tags,P):
			if D<H:continue
			if B.startswith('rating:'):continue
			C[B]=D
		if K:F=sorted(C)
		else:F=[A for(A,B)in sorted(C.items(),key=lambda x:-x[1])]
		G=[];Q={A.strip().replace(' ','_')for A in shared.opts.deepbooru_filter_tags.split(',')}
		for B in[A for A in F if A not in Q]:
			D=C[B];A=B
			if I:A=A.replace('_',' ')
			if J:A=re.sub(re_special,'\\\\\\1',A)
			if L:A=f"({A}:{D:.3f})"
			G.append(A)
		return', '.join(G)
model=DeepDanbooru()