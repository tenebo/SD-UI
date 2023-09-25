_L='embedding'
_K='optimizer_state_dict'
_J='cpu'
_I='sd-ti-embedding'
_H='sd_checkpoint_name'
_G='sd_checkpoint'
_F='step'
_E='string_to_param'
_D='name'
_C=False
_B=True
_A=None
import os
from collections import namedtuple
from contextlib import closing
import torch,tqdm,html,datetime,csv,safetensors.torch,numpy as np
from PIL import Image,PngImagePlugin
from torch.utils.tensorboard import SummaryWriter
from modules import shared,devices,sd_hijack,sd_models,images,sd_samplers,sd_hijack_checkpoint,errors,hashes
import modules.textual_inversion.dataset
from modules.textual_inversion.learn_schedule import LearnRateScheduler
from modules.textual_inversion.image_embedding import embedding_to_b64,embedding_from_b64,insert_image_data_embed,extract_image_data_embed,caption_image_overlay
from modules.textual_inversion.logging import save_settings_to_file
TextualInversionTemplate=namedtuple('TextualInversionTemplate',[_D,'path'])
textual_inversion_templates={}
def list_textual_inversion_templates():
	textual_inversion_templates.clear()
	for(root,_,fns)in os.walk(shared.cmd_opts.textual_inversion_templates_dir):
		for fn in fns:path=os.path.join(root,fn);textual_inversion_templates[fn]=TextualInversionTemplate(fn,path)
	return textual_inversion_templates
class Embedding:
	def __init__(self,vec,name,step=_A):self.vec=vec;self.name=name;self.step=step;self.shape=_A;self.vectors=0;self.cached_checksum=_A;self.sd_checkpoint=_A;self.sd_checkpoint_name=_A;self.optimizer_state_dict=_A;self.filename=_A;self.hash=_A;self.shorthash=_A
	def save(self,filename):
		embedding_data={'string_to_token':{'*':265},_E:{'*':self.vec},_D:self.name,_F:self.step,_G:self.sd_checkpoint,_H:self.sd_checkpoint_name};torch.save(embedding_data,filename)
		if shared.opts.save_optimizer_state and self.optimizer_state_dict is not _A:optimizer_saved_dict={'hash':self.checksum(),_K:self.optimizer_state_dict};torch.save(optimizer_saved_dict,f"{filename}.optim")
	def checksum(self):
		if self.cached_checksum is not _A:return self.cached_checksum
		def const_hash(a):
			r=0
			for v in a:r=(r*281^int(v)*997)&4294967295
			return r
		self.cached_checksum=f"{const_hash(self.vec.reshape(-1)*100)&65535:04x}";return self.cached_checksum
	def set_hash(self,v):self.hash=v;self.shorthash=self.hash[0:12]
class DirWithTextualInversionEmbeddings:
	def __init__(self,path):self.path=path;self.mtime=_A
	def has_changed(self):
		if not os.path.isdir(self.path):return _C
		mt=os.path.getmtime(self.path)
		if self.mtime is _A or mt>self.mtime:return _B
	def update(self):
		if not os.path.isdir(self.path):return
		self.mtime=os.path.getmtime(self.path)
class EmbeddingDatabase:
	def __init__(self):self.ids_lookup={};self.word_embeddings={};self.skipped_embeddings={};self.expected_shape=-1;self.embedding_dirs={};self.previously_displayed_embeddings=()
	def add_embedding_dir(self,path):self.embedding_dirs[path]=DirWithTextualInversionEmbeddings(path)
	def clear_embedding_dirs(self):self.embedding_dirs.clear()
	def register_embedding(self,embedding,model):return self.register_embedding_by_name(embedding,model,embedding.name)
	def register_embedding_by_name(self,embedding,model,name):
		ids=model.cond_stage_model.tokenize([name])[0];first_id=ids[0]
		if first_id not in self.ids_lookup:self.ids_lookup[first_id]=[]
		if name in self.word_embeddings:lookup=[x for x in self.ids_lookup[first_id]if x[1].name!=name]
		else:lookup=self.ids_lookup[first_id]
		if embedding is not _A:lookup+=[(ids,embedding)]
		self.ids_lookup[first_id]=sorted(lookup,key=lambda x:len(x[0]),reverse=_B)
		if embedding is _A:
			if name in self.word_embeddings:del self.word_embeddings[name]
			if len(self.ids_lookup[first_id])==0:del self.ids_lookup[first_id]
			return
		self.word_embeddings[name]=embedding;return embedding
	def get_expected_shape(self):vec=shared.sd_model.cond_stage_model.encode_embedding_init_text(',',1);return vec.shape[1]
	def load_from_file(self,path,filename):
		C='clip_l';B='embedding file has multiple terms in it';A='clip_g';name,ext=os.path.splitext(filename);ext=ext.upper()
		if ext in['.PNG','.WEBP','.JXL','.AVIF']:
			_,second_ext=os.path.splitext(name)
			if second_ext.upper()=='.PREVIEW':return
			embed_image=Image.open(path)
			if hasattr(embed_image,'text')and _I in embed_image.text:data=embedding_from_b64(embed_image.text[_I]);name=data.get(_D,name)
			else:
				data=extract_image_data_embed(embed_image)
				if data:name=data.get(_D,name)
				else:return
		elif ext in['.BIN','.PT']:data=torch.load(path,map_location=_J)
		elif ext in['.SAFETENSORS']:data=safetensors.torch.load_file(path,device=_J)
		else:return
		if _E in data:param_dict=data[_E];param_dict=getattr(param_dict,'_parameters',param_dict);assert len(param_dict)==1,B;emb=next(iter(param_dict.items()))[1];vec=emb.detach().to(devices.device,dtype=torch.float32);shape=vec.shape[-1];vectors=vec.shape[0]
		elif type(data)==dict and A in data and C in data:vec={k:v.detach().to(devices.device,dtype=torch.float32)for(k,v)in data.items()};shape=data[A].shape[-1]+data[C].shape[-1];vectors=data[A].shape[0]
		elif type(data)==dict and type(next(iter(data.values())))==torch.Tensor:
			assert len(data.keys())==1,B;emb=next(iter(data.values()))
			if len(emb.shape)==1:emb=emb.unsqueeze(0)
			vec=emb.detach().to(devices.device,dtype=torch.float32);shape=vec.shape[-1];vectors=vec.shape[0]
		else:raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")
		embedding=Embedding(vec,name);embedding.step=data.get(_F,_A);embedding.sd_checkpoint=data.get(_G,_A);embedding.sd_checkpoint_name=data.get(_H,_A);embedding.vectors=vectors;embedding.shape=shape;embedding.filename=path;embedding.set_hash(hashes.sha256(embedding.filename,'textual_inversion/'+name)or'')
		if self.expected_shape==-1 or self.expected_shape==embedding.shape:self.register_embedding(embedding,shared.sd_model)
		else:self.skipped_embeddings[name]=embedding
	def load_from_dir(self,embdir):
		if not os.path.isdir(embdir.path):return
		for(root,_,fns)in os.walk(embdir.path,followlinks=_B):
			for fn in fns:
				try:
					fullfn=os.path.join(root,fn)
					if os.stat(fullfn).st_size==0:continue
					self.load_from_file(fullfn,fn)
				except Exception:errors.report(f"Error loading embedding {fn}",exc_info=_B);continue
	def load_textual_inversion_embeddings(self,force_reload=_C):
		if not force_reload:
			need_reload=_C
			for embdir in self.embedding_dirs.values():
				if embdir.has_changed():need_reload=_B;break
			if not need_reload:return
		self.ids_lookup.clear();self.word_embeddings.clear();self.skipped_embeddings.clear();self.expected_shape=self.get_expected_shape()
		for embdir in self.embedding_dirs.values():self.load_from_dir(embdir);embdir.update()
		sorted_word_embeddings={e.name:e for e in sorted(self.word_embeddings.values(),key=lambda e:e.name.lower())};self.word_embeddings.clear();self.word_embeddings.update(sorted_word_embeddings);displayed_embeddings=tuple(self.word_embeddings.keys()),tuple(self.skipped_embeddings.keys())
		if shared.opts.textual_inversion_print_at_load and self.previously_displayed_embeddings!=displayed_embeddings:
			self.previously_displayed_embeddings=displayed_embeddings;print(f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}")
			if self.skipped_embeddings:print(f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}")
	def find_embedding_at_position(self,tokens,offset):
		token=tokens[offset];possible_matches=self.ids_lookup.get(token,_A)
		if possible_matches is _A:return _A,_A
		for(ids,embedding)in possible_matches:
			if tokens[offset:offset+len(ids)]==ids:return embedding,len(ids)
		return _A,_A
def create_embedding(name,num_vectors_per_token,overwrite_old,init_text='*'):
	cond_model=shared.sd_model.cond_stage_model
	with devices.autocast():cond_model([''])
	embedded=cond_model.encode_embedding_init_text(init_text or'*',num_vectors_per_token);vec=torch.zeros((num_vectors_per_token,embedded.shape[1]),device=devices.device)
	if init_text:
		for i in range(num_vectors_per_token):vec[i]=embedded[i*int(embedded.shape[0])//num_vectors_per_token]
	name=''.join(x for x in name if x.isalnum()or x in'._- ');fn=os.path.join(shared.cmd_opts.embeddings_dir,f"{name}.pt")
	if not overwrite_old:assert not os.path.exists(fn),f"file {fn} already exists"
	embedding=Embedding(vec,name);embedding.step=0;embedding.save(fn);return fn
def write_loss(log_directory,filename,step,epoch_len,values):
	B='epoch_step';A='epoch'
	if shared.opts.training_write_csv_every==0:return
	if step%shared.opts.training_write_csv_every!=0:return
	write_csv_header=_C if os.path.exists(os.path.join(log_directory,filename))else _B
	with open(os.path.join(log_directory,filename),'a+',newline='')as fout:
		csv_writer=csv.DictWriter(fout,fieldnames=[_F,A,B,*values.keys()])
		if write_csv_header:csv_writer.writeheader()
		epoch=(step-1)//epoch_len;epoch_step=(step-1)%epoch_len;csv_writer.writerow({_F:step,A:epoch,B:epoch_step,**values})
def tensorboard_setup(log_directory):A='tensorboard';os.makedirs(os.path.join(log_directory,A),exist_ok=_B);return SummaryWriter(log_dir=os.path.join(log_directory,A),flush_secs=shared.opts.training_tensorboard_flush_every)
def tensorboard_add(tensorboard_writer,loss,global_step,step,learn_rate,epoch_num):tensorboard_add_scaler(tensorboard_writer,'Loss/train',loss,global_step);tensorboard_add_scaler(tensorboard_writer,f"Loss/train/epoch-{epoch_num}",loss,step);tensorboard_add_scaler(tensorboard_writer,'Learn rate/train',learn_rate,global_step);tensorboard_add_scaler(tensorboard_writer,f"Learn rate/train/epoch-{epoch_num}",learn_rate,step)
def tensorboard_add_scaler(tensorboard_writer,tag,value,step):tensorboard_writer.add_scalar(tag=tag,scalar_value=value,global_step=step)
def tensorboard_add_image(tensorboard_writer,tag,pil_image,step):img_tensor=torch.as_tensor(np.array(pil_image,copy=_B));img_tensor=img_tensor.view(pil_image.size[1],pil_image.size[0],len(pil_image.getbands()));img_tensor=img_tensor.permute((2,0,1));tensorboard_writer.add_image(tag,img_tensor,global_step=step)
def validate_train_inputs(model_name,learn_rate,batch_size,gradient_step,data_root,template_file,template_filename,steps,save_model_every,create_image_every,log_directory,name=_L):
	A='Dataset directory is empty';assert model_name,f"{name} not selected";assert learn_rate,'Learning rate is empty or 0';assert isinstance(batch_size,int),'Batch size must be integer';assert batch_size>0,'Batch size must be positive';assert isinstance(gradient_step,int),'Gradient accumulation step must be integer';assert gradient_step>0,'Gradient accumulation step must be positive';assert data_root,A;assert os.path.isdir(data_root),"Dataset directory doesn't exist";assert os.listdir(data_root),A;assert template_filename,'Prompt template file not selected';assert template_file,f"Prompt template file {template_filename} not found";assert os.path.isfile(template_file.path),f"Prompt template file {template_filename} doesn't exist";assert steps,'Max steps is empty or 0';assert isinstance(steps,int),'Max steps must be integer';assert steps>0,'Max steps must be positive';assert isinstance(save_model_every,int),'Save {name} must be integer';assert save_model_every>=0,'Save {name} must be positive or 0';assert isinstance(create_image_every,int),'Create image must be integer';assert create_image_every>=0,'Create image must be positive or 0'
	if save_model_every or create_image_every:assert log_directory,'Log directory is empty'
def train_embedding(id_task,embedding_name,learn_rate,batch_size,gradient_step,data_root,log_directory,training_width,training_height,varsize,steps,clip_grad_mode,clip_grad_value,shuffle_tags,tag_drop_out,latent_sampling_method,use_weight,create_image_every,save_embedding_every,template_filename,save_image_with_stored_embedding,preview_from_txt2img,preview_prompt,preview_negative_prompt,preview_steps,preview_sampler_index,preview_cfg_scale,preview_seed,preview_width,preview_height):
	A='<none>';from modules import processing;save_embedding_every=save_embedding_every or 0;create_image_every=create_image_every or 0;template_file=textual_inversion_templates.get(template_filename,_A);validate_train_inputs(embedding_name,learn_rate,batch_size,gradient_step,data_root,template_file,template_filename,steps,save_embedding_every,create_image_every,log_directory,name=_L);template_file=template_file.path;shared.state.job='train-embedding';shared.state.textinfo='Initializing textual inversion training...';shared.state.job_count=steps;filename=os.path.join(shared.cmd_opts.embeddings_dir,f"{embedding_name}.pt");log_directory=os.path.join(log_directory,datetime.datetime.now().strftime('%Y-%m-%d'),embedding_name);unload=shared.opts.unload_models_when_training
	if save_embedding_every>0:embedding_dir=os.path.join(log_directory,'embeddings');os.makedirs(embedding_dir,exist_ok=_B)
	else:embedding_dir=_A
	if create_image_every>0:images_dir=os.path.join(log_directory,'images');os.makedirs(images_dir,exist_ok=_B)
	else:images_dir=_A
	if create_image_every>0 and save_image_with_stored_embedding:images_embeds_dir=os.path.join(log_directory,'image_embeddings');os.makedirs(images_embeds_dir,exist_ok=_B)
	else:images_embeds_dir=_A
	hijack=sd_hijack.model_hijack;embedding=hijack.embedding_db.word_embeddings[embedding_name];checkpoint=sd_models.select_checkpoint();initial_step=embedding.step or 0
	if initial_step>=steps:shared.state.textinfo='Model has already been trained beyond specified max steps';return embedding,filename
	scheduler=LearnRateScheduler(learn_rate,steps,initial_step);clip_grad=torch.nn.utils.clip_grad_value_ if clip_grad_mode=='value'else torch.nn.utils.clip_grad_norm_ if clip_grad_mode=='norm'else _A
	if clip_grad:clip_grad_sched=LearnRateScheduler(clip_grad_value,steps,initial_step,verbose=_C)
	shared.state.textinfo=f"Preparing dataset from {html.escape(data_root)}...";old_parallel_processing_allowed=shared.parallel_processing_allowed
	if shared.opts.training_enable_tensorboard:tensorboard_writer=tensorboard_setup(log_directory)
	pin_memory=shared.opts.pin_memory;ds=modules.textual_inversion.dataset.PersonalizedBase(data_root=data_root,width=training_width,height=training_height,repeats=shared.opts.training_image_repeats_per_epoch,placeholder_token=embedding_name,model=shared.sd_model,cond_model=shared.sd_model.cond_stage_model,device=devices.device,template_file=template_file,batch_size=batch_size,gradient_step=gradient_step,shuffle_tags=shuffle_tags,tag_drop_out=tag_drop_out,latent_sampling_method=latent_sampling_method,varsize=varsize,use_weight=use_weight)
	if shared.opts.save_training_settings_to_txt:save_settings_to_file(log_directory,{**dict(model_name=checkpoint.model_name,model_hash=checkpoint.shorthash,num_of_dataset_images=len(ds),num_vectors_per_token=len(embedding.vec)),**locals()})
	latent_sampling_method=ds.latent_sampling_method;dl=modules.textual_inversion.dataset.PersonalizedDataLoader(ds,latent_sampling_method=latent_sampling_method,batch_size=ds.batch_size,pin_memory=pin_memory)
	if unload:shared.parallel_processing_allowed=_C;shared.sd_model.first_stage_model.to(devices.cpu)
	embedding.vec.requires_grad=_B;optimizer=torch.optim.AdamW([embedding.vec],lr=scheduler.learn_rate,weight_decay=.0)
	if shared.opts.save_optimizer_state:
		optimizer_state_dict=_A
		if os.path.exists(f"{filename}.optim"):
			optimizer_saved_dict=torch.load(f"{filename}.optim",map_location=_J)
			if embedding.checksum()==optimizer_saved_dict.get('hash',_A):optimizer_state_dict=optimizer_saved_dict.get(_K,_A)
		if optimizer_state_dict is not _A:optimizer.load_state_dict(optimizer_state_dict);print('Loaded existing optimizer from checkpoint')
		else:print('No saved optimizer exists in checkpoint')
	scaler=torch.cuda.amp.GradScaler();batch_size=ds.batch_size;gradient_step=ds.gradient_step;steps_per_epoch=len(ds)//batch_size//gradient_step;max_steps_per_epoch=len(ds)//batch_size-len(ds)//batch_size%gradient_step;loss_step=0;_loss_step=0;last_saved_file=A;last_saved_image=A;forced_filename=A;embedding_yet_to_be_embedded=_C;is_training_inpainting_model=shared.sd_model.model.conditioning_key in{'hybrid','concat'};img_c=_A;pbar=tqdm.tqdm(total=steps-initial_step)
	try:
		sd_hijack_checkpoint.add()
		for _ in range((steps-initial_step)*gradient_step):
			if scheduler.finished:break
			if shared.state.interrupted:break
			for(j,batch)in enumerate(dl):
				if j==max_steps_per_epoch:break
				scheduler.apply(optimizer,embedding.step)
				if scheduler.finished:break
				if shared.state.interrupted:break
				if clip_grad:clip_grad_sched.step(embedding.step)
				with devices.autocast():
					x=batch.latent_sample.to(devices.device,non_blocking=pin_memory)
					if use_weight:w=batch.weight.to(devices.device,non_blocking=pin_memory)
					c=shared.sd_model.cond_stage_model(batch.cond_text)
					if is_training_inpainting_model:
						if img_c is _A:img_c=processing.txt2img_image_conditioning(shared.sd_model,c,training_width,training_height)
						cond={'c_concat':[img_c],'c_crossattn':[c]}
					else:cond=c
					if use_weight:loss=shared.sd_model.weighted_forward(x,cond,w)[0]/gradient_step;del w
					else:loss=shared.sd_model.forward(x,cond)[0]/gradient_step
					del x;_loss_step+=loss.item()
				scaler.scale(loss).backward()
				if(j+1)%gradient_step!=0:continue
				if clip_grad:clip_grad(embedding.vec,clip_grad_sched.learn_rate)
				scaler.step(optimizer);scaler.update();embedding.step+=1;pbar.update();optimizer.zero_grad(set_to_none=_B);loss_step=_loss_step;_loss_step=0;steps_done=embedding.step+1;epoch_num=embedding.step//steps_per_epoch;epoch_step=embedding.step%steps_per_epoch;description=f"Training textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}";pbar.set_description(description)
				if embedding_dir is not _A and steps_done%save_embedding_every==0:embedding_name_every=f"{embedding_name}-{steps_done}";last_saved_file=os.path.join(embedding_dir,f"{embedding_name_every}.pt");save_embedding(embedding,optimizer,checkpoint,embedding_name_every,last_saved_file,remove_cached_checksum=_B);embedding_yet_to_be_embedded=_B
				write_loss(log_directory,'textual_inversion_loss.csv',embedding.step,steps_per_epoch,{'loss':f"{loss_step:.7f}",'learn_rate':scheduler.learn_rate})
				if images_dir is not _A and steps_done%create_image_every==0:
					forced_filename=f"{embedding_name}-{steps_done}";last_saved_image=os.path.join(images_dir,forced_filename);shared.sd_model.first_stage_model.to(devices.device);p=processing.StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model,do_not_save_grid=_B,do_not_save_samples=_B,do_not_reload_embeddings=_B)
					if preview_from_txt2img:p.prompt=preview_prompt;p.negative_prompt=preview_negative_prompt;p.steps=preview_steps;p.sampler_name=sd_samplers.samplers[preview_sampler_index].name;p.cfg_scale=preview_cfg_scale;p.seed=preview_seed;p.width=preview_width;p.height=preview_height
					else:p.prompt=batch.cond_text[0];p.steps=20;p.width=training_width;p.height=training_height
					preview_text=p.prompt
					with closing(p):processed=processing.process_images(p);image=processed.images[0]if len(processed.images)>0 else _A
					if unload:shared.sd_model.first_stage_model.to(devices.cpu)
					if image is not _A:
						shared.state.assign_current_image(image);last_saved_image,last_text_info=images.save_image(image,images_dir,'',p.seed,p.prompt,shared.opts.samples_format,processed.infotexts[0],p=p,forced_filename=forced_filename,save_to_dirs=_C);last_saved_image+=f", prompt: {preview_text}"
						if shared.opts.training_enable_tensorboard and shared.opts.training_tensorboard_save_images:tensorboard_add_image(tensorboard_writer,f"Validation at epoch {epoch_num}",image,embedding.step)
					if save_image_with_stored_embedding and os.path.exists(last_saved_file)and embedding_yet_to_be_embedded:
						last_saved_image_chunks=os.path.join(images_embeds_dir,f"{embedding_name}-{steps_done}.png");info=PngImagePlugin.PngInfo();data=torch.load(last_saved_file);info.add_text(_I,embedding_to_b64(data));title=f"<{data.get(_D,'???')}>"
						try:vectorSize=list(data[_E].values())[0].shape[0]
						except Exception:vectorSize='?'
						checkpoint=sd_models.select_checkpoint();footer_left=checkpoint.model_name;footer_mid=f"[{checkpoint.shorthash}]";footer_right=f"{vectorSize}v {steps_done}s";captioned_image=caption_image_overlay(image,title,footer_left,footer_mid,footer_right);captioned_image=insert_image_data_embed(captioned_image,data);captioned_image.save(last_saved_image_chunks,'PNG',pnginfo=info);embedding_yet_to_be_embedded=_C
					last_saved_image,last_text_info=images.save_image(image,images_dir,'',p.seed,p.prompt,shared.opts.samples_format,processed.infotexts[0],p=p,forced_filename=forced_filename,save_to_dirs=_C);last_saved_image+=f", prompt: {preview_text}"
				shared.state.job_no=embedding.step;shared.state.textinfo=f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(batch.cond_text[0])}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""
		filename=os.path.join(shared.cmd_opts.embeddings_dir,f"{embedding_name}.pt");save_embedding(embedding,optimizer,checkpoint,embedding_name,filename,remove_cached_checksum=_B)
	except Exception:errors.report('Error training embedding',exc_info=_B)
	finally:pbar.leave=_C;pbar.close();shared.sd_model.first_stage_model.to(devices.device);shared.parallel_processing_allowed=old_parallel_processing_allowed;sd_hijack_checkpoint.remove()
	return embedding,filename
def save_embedding(embedding,optimizer,checkpoint,embedding_name,filename,remove_cached_checksum=_B):
	old_embedding_name=embedding.name;old_sd_checkpoint=embedding.sd_checkpoint if hasattr(embedding,_G)else _A;old_sd_checkpoint_name=embedding.sd_checkpoint_name if hasattr(embedding,_H)else _A;old_cached_checksum=embedding.cached_checksum if hasattr(embedding,'cached_checksum')else _A
	try:
		embedding.sd_checkpoint=checkpoint.shorthash;embedding.sd_checkpoint_name=checkpoint.model_name
		if remove_cached_checksum:embedding.cached_checksum=_A
		embedding.name=embedding_name;embedding.optimizer_state_dict=optimizer.state_dict();embedding.save(filename)
	except:embedding.sd_checkpoint=old_sd_checkpoint;embedding.sd_checkpoint_name=old_sd_checkpoint_name;embedding.name=old_embedding_name;embedding.cached_checksum=old_cached_checksum;raise