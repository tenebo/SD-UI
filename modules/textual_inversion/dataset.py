_D='random'
_C='once'
_B=False
_A=None
import os,numpy as np,PIL,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader,Sampler
from torchvision import transforms
from collections import defaultdict
from random import shuffle,choices
import random,tqdm
from modules import devices,shared
import re
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
re_numbers_at_start=re.compile('^[-\\d]+\\s*')
class DatasetEntry:
	def __init__(A,filename=_A,filename_text=_A,latent_dist=_A,latent_sample=_A,cond=_A,cond_text=_A,pixel_values=_A,weight=_A):A.filename=filename;A.filename_text=filename_text;A.weight=weight;A.latent_dist=latent_dist;A.latent_sample=latent_sample;A.cond=cond;A.cond_text=cond_text;A.pixel_values=pixel_values
class PersonalizedBase(Dataset):
	def __init__(A,data_root,width,height,repeats,flip_p=.5,placeholder_token='*',model=_A,cond_model=_A,device=_A,template_file=_A,include_cond=_B,batch_size=1,gradient_step=1,shuffle_tags=_B,tag_drop_out=0,latent_sampling_method=_C,varsize=_B,use_weight=_B):
		P=model;L=use_weight;J=latent_sampling_method;E=data_root;Q=re.compile(shared.opts.dataset_filename_word_regex)if shared.opts.dataset_filename_word_regex else _A;A.placeholder_token=placeholder_token;A.flip=transforms.RandomHorizontalFlip(p=flip_p);A.dataset=[]
		with open(template_file,'r')as M:V=[A.strip()for A in M.readlines()]
		A.lines=V;assert E,'dataset directory not specified';assert os.path.isdir(E),"Dataset directory doesn't exist";assert os.listdir(E),'Dataset directory is empty';A.image_paths=[os.path.join(E,A)for A in os.listdir(E)];A.shuffle_tags=shuffle_tags;A.tag_drop_out=tag_drop_out;K=defaultdict(list);print('Preparing dataset...')
		for F in tqdm.tqdm(A.image_paths):
			N=_A
			if shared.state.interrupted:raise Exception('interrupted')
			try:
				C=Image.open(F)
				if L and'A'in C.getbands():N=C.getchannel('A')
				C=C.convert('RGB')
				if not varsize:C=C.resize((width,height),PIL.Image.BICUBIC)
			except Exception:continue
			R=f"{os.path.splitext(F)[0]}.txt";W=os.path.basename(F)
			if os.path.exists(R):
				with open(R,'r',encoding='utf8')as M:D=M.read()
			else:
				D=os.path.splitext(W)[0];D=re.sub(re_numbers_at_start,'',D)
				if Q:X=Q.findall(D);D=(shared.opts.dataset_filename_join_string or'').join(X)
			O=np.array(C).astype(np.uint8);O=(O/127.5-1.).astype(np.float32);S=torch.from_numpy(O).permute(2,0,1).to(device=device,dtype=torch.float32);G=_A
			with devices.autocast():H=P.encode_first_stage(S.unsqueeze(dim=0))
			if J=='deterministic':
				if isinstance(H,DiagonalGaussianDistribution):H.std=0
				else:J=_C
			G=P.get_first_stage_encoding(H).squeeze().to(devices.cpu)
			if L and N is not _A:T,*U=G.shape;Y=N.resize(U);Z=np.array(Y).astype(np.float32);B=torch.tensor([Z]*T).reshape([T]+U);B-=B.min();B/=B.mean()
			elif L:B=torch.ones(G.shape)
			else:B=_A
			if J==_D:I=DatasetEntry(filename=F,filename_text=D,latent_dist=H,weight=B)
			else:I=DatasetEntry(filename=F,filename_text=D,latent_sample=G,weight=B)
			if not(A.tag_drop_out!=0 or A.shuffle_tags):I.cond_text=A.create_text(D)
			if include_cond and not(A.tag_drop_out!=0 or A.shuffle_tags):
				with devices.autocast():I.cond=cond_model([I.cond_text]).to(devices.cpu).squeeze(0)
			K[C.size].append(len(A.dataset));A.dataset.append(I);del S;del H;del G;del B
		A.length=len(A.dataset);A.groups=list(K.values());assert A.length>0,'No images have been found in the dataset.';A.batch_size=min(batch_size,A.length);A.gradient_step=min(gradient_step,A.length//A.batch_size);A.latent_sampling_method=J
		if len(K)>1:
			print('Buckets:')
			for((a,b),c)in sorted(K.items(),key=lambda x:x[0]):print(f"  {a}x{b}: {len(c)}")
			print()
	def create_text(A,filename_text):
		B=random.choice(A.lines);C=filename_text.split(',')
		if A.tag_drop_out!=0:C=[B for B in C if random.random()>A.tag_drop_out]
		if A.shuffle_tags:random.shuffle(C)
		B=B.replace('[filewords]',','.join(C));B=B.replace('[name]',A.placeholder_token);return B
	def __len__(A):return A.length
	def __getitem__(A,i):
		B=A.dataset[i]
		if A.tag_drop_out!=0 or A.shuffle_tags:B.cond_text=A.create_text(B.filename_text)
		if A.latent_sampling_method==_D:B.latent_sample=shared.sd_model.get_first_stage_encoding(B.latent_dist).to(devices.cpu)
		return B
class GroupedBatchSampler(Sampler):
	def __init__(A,data_source,batch_size):C=data_source;B=batch_size;super().__init__(C);D=len(C);A.groups=C.groups;A.len=E=D//B;F=[len(A)/D*E*B for A in C.groups];A.base=[int(A)//B for A in F];A.n_rand_batches=G=E-sum(A.base);A.probs=[A%B/G/B if G>0 else 0 for A in F];A.batch_size=B
	def __len__(A):return A.len
	def __iter__(A):
		B=A.batch_size
		for C in A.groups:shuffle(C)
		D=[]
		for C in A.groups:D.extend(C[A*B:(A+1)*B]for A in range(len(C)//B))
		for F in range(A.n_rand_batches):E=choices(A.groups,A.probs)[0];D.append(choices(E,k=B))
		shuffle(D);yield from D
class PersonalizedDataLoader(DataLoader):
	def __init__(A,dataset,latent_sampling_method=_C,batch_size=1,pin_memory=_B):
		B=dataset;super(PersonalizedDataLoader,A).__init__(B,batch_sampler=GroupedBatchSampler(B,batch_size),pin_memory=pin_memory)
		if latent_sampling_method==_D:A.collate_fn=collate_wrapper_random
		else:A.collate_fn=collate_wrapper
class BatchLoader:
	def __init__(A,data):
		B=data;A.cond_text=[A.cond_text for A in B];A.cond=[A.cond for A in B];A.latent_sample=torch.stack([A.latent_sample for A in B]).squeeze(1)
		if all(A.weight is not _A for A in B):A.weight=torch.stack([A.weight for A in B]).squeeze(1)
		else:A.weight=_A
	def pin_memory(A):A.latent_sample=A.latent_sample.pin_memory();return A
def collate_wrapper(batch):return BatchLoader(batch)
class BatchLoaderRandom(BatchLoader):
	def __init__(A,data):super().__init__(data)
	def pin_memory(A):return A
def collate_wrapper_random(batch):return BatchLoaderRandom(batch)