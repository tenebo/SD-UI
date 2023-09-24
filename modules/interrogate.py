_B=True
_A=None
import os,sys
from collections import namedtuple
from pathlib import Path
import re,torch,torch.hub
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules import devices,paths,shared,lowvram,modelloader,errors
blip_image_eval_size=384
clip_model_name='ViT-L/14'
Category=namedtuple('Category',['name','topn','items'])
re_topn=re.compile('\\.top(\\d+)\\.')
def category_types():return[A.stem for A in Path(shared.interrogator.content_dir).glob('*.txt')]
def download_default_clip_interrogate_categories(content_dir):
	B=content_dir;print('Downloading CLIP categories...');A=f"{B}_tmp";D=['artists','flavors','mediums','movements']
	try:
		os.makedirs(A,exist_ok=_B)
		for C in D:torch.hub.download_url_to_file(f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{C}.txt",os.path.join(A,f"{C}.txt"))
		os.rename(A,B)
	except Exception as E:errors.display(E,'downloading default CLIP interrogate categories')
	finally:
		if os.path.exists(A):os.removedirs(A)
class InterrogateModels:
	blip_model=_A;clip_model=_A;clip_preprocess=_A;dtype=_A;running_on_cpu=_A
	def __init__(A,content_dir):A.loaded_categories=_A;A.skip_categories=[];A.content_dir=content_dir;A.running_on_cpu=devices.device_interrogate==torch.device('cpu')
	def categories(A):
		if not os.path.exists(A.content_dir):download_default_clip_interrogate_categories(A.content_dir)
		if A.loaded_categories is not _A and A.skip_categories==shared.opts.interrogate_clip_skip_categories:return A.loaded_categories
		A.loaded_categories=[]
		if os.path.exists(A.content_dir):
			A.skip_categories=shared.opts.interrogate_clip_skip_categories;D=[]
			for B in Path(A.content_dir).glob('*.txt'):
				D.append(B.stem)
				if B.stem in A.skip_categories:continue
				C=re_topn.search(B.stem);E=1 if C is _A else int(C.group(1))
				with open(B,'r',encoding='utf8')as F:G=[A.strip()for A in F.readlines()]
				A.loaded_categories.append(Category(name=B.stem,topn=E,items=G))
		return A.loaded_categories
	def create_fake_fairscale(B):
		class A:
			def checkpoint_wrapper(A):0
		sys.modules['fairscale.nn.checkpoint.checkpoint_activations']=A
	def load_blip_model(C):B='BLIP';C.create_fake_fairscale();import models.blip;D=modelloader.load_models(model_path=os.path.join(paths.models_path,B),model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',ext_filter=['.pth'],download_name='model_base_caption_capfilt_large.pth');A=models.blip.blip_decoder(pretrained=D[0],image_size=blip_image_eval_size,vit='base',med_config=os.path.join(paths.paths[B],'configs','med_config.json'));A.eval();return A
	def load_clip_model(D):
		import clip as B
		if D.running_on_cpu:A,C=B.load(clip_model_name,device='cpu',download_root=shared.cmd_opts.clip_models_path)
		else:A,C=B.load(clip_model_name,download_root=shared.cmd_opts.clip_models_path)
		A.eval();A=A.to(devices.device_interrogate);return A,C
	def load(A):
		if A.blip_model is _A:
			A.blip_model=A.load_blip_model()
			if not shared.cmd_opts.no_half and not A.running_on_cpu:A.blip_model=A.blip_model.half()
		A.blip_model=A.blip_model.to(devices.device_interrogate)
		if A.clip_model is _A:
			A.clip_model,A.clip_preprocess=A.load_clip_model()
			if not shared.cmd_opts.no_half and not A.running_on_cpu:A.clip_model=A.clip_model.half()
		A.clip_model=A.clip_model.to(devices.device_interrogate);A.dtype=next(A.clip_model.parameters()).dtype
	def send_clip_to_ram(A):
		if not shared.opts.interrogate_keep_models_in_memory:
			if A.clip_model is not _A:A.clip_model=A.clip_model.to(devices.cpu)
	def send_blip_to_ram(A):
		if not shared.opts.interrogate_keep_models_in_memory:
			if A.blip_model is not _A:A.blip_model=A.blip_model.to(devices.cpu)
	def unload(A):A.send_clip_to_ram();A.send_blip_to_ram();devices.torch_gc()
	def rank(F,image_features,text_array,top_count=1):
		C=image_features;B=top_count;A=text_array;import clip;devices.torch_gc()
		if shared.opts.interrogate_clip_dict_limit!=0:A=A[0:int(shared.opts.interrogate_clip_dict_limit)]
		B=min(B,len(A));G=clip.tokenize(list(A),truncate=_B).to(devices.device_interrogate);D=F.clip_model.encode_text(G).type(F.dtype);D/=D.norm(dim=-1,keepdim=_B);E=torch.zeros((1,len(A))).to(devices.device_interrogate)
		for H in range(C.shape[0]):E+=(1e2*C[H].unsqueeze(0)@D.T).softmax(dim=-1)
		E/=C.shape[0];I,J=E.cpu().topk(B,dim=-1);return[(A[J[0][B].numpy()],I[0][B].numpy()*100)for B in range(B)]
	def generate_caption(A,pil_image):
		B=transforms.Compose([transforms.Resize((blip_image_eval_size,blip_image_eval_size),interpolation=InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])(pil_image).unsqueeze(0).type(A.dtype).to(devices.device_interrogate)
		with torch.no_grad():C=A.blip_model.generate(B,sample=False,num_beams=shared.opts.interrogate_clip_num_beams,min_length=shared.opts.interrogate_clip_min_length,max_length=shared.opts.interrogate_clip_max_length)
		return C[0]
	def interrogate(A,pil_image):
		D=pil_image;B='';shared.state.begin(job='interrogate')
		try:
			lowvram.send_everything_to_cpu();devices.torch_gc();A.load();G=A.generate_caption(D);A.send_blip_to_ram();devices.torch_gc();B=G;H=A.clip_preprocess(D).unsqueeze(0).type(A.dtype).to(devices.device_interrogate)
			with torch.no_grad(),devices.autocast():
				C=A.clip_model.encode_image(H).type(A.dtype);C/=C.norm(dim=-1,keepdim=_B)
				for E in A.categories():
					I=A.rank(C,E.items,top_count=E.topn)
					for(F,J)in I:
						if shared.opts.interrogate_return_ranks:B+=f", ({F}:{J/100:.3f})"
						else:B+=f", {F}"
		except Exception:errors.report('Error interrogating',exc_info=_B);B+='<error>'
		A.unload();shared.state.end();return B