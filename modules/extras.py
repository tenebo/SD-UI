_B=False
_A=None
import os,re,shutil,json,torch,tqdm
from modules import shared,images,sd_models,sd_vae,sd_models_config,errors
from modules.ui_common import plaintext_to_html
import gradio as gr,safetensors.torch
def run_pnginfo(image):
	C=image
	if C is _A:return'','',''
	D,B=images.read_info_from_image(C);B={**{'parameters':D},**B};A=''
	for(E,F)in B.items():A+=f"""
<div>
<p><b>{plaintext_to_html(str(E))}</b></p>
<p>{plaintext_to_html(str(F))}</p>
</div>
""".strip()+'\n'
	if len(A)==0:G='Nothing found in the image.';A=f"<div><p>{G}<p></div>"
	return'',D,A
def create_config(ckpt_result,config_source,a,b,c):
	C=config_source
	def B(x):A=sd_models_config.find_checkpoint_config_near_filename(x)if x else _A;return A if A!=shared.sd_default_config else _A
	if C==0:A=B(a)or B(b)or B(c)
	elif C==1:A=B(b)
	elif C==2:A=B(c)
	else:A=_A
	if A is _A:return
	E,F=os.path.splitext(ckpt_result);D=E+'.yaml';print('Copying config:');print('   from:',A);print('     to:',D);shutil.copyfile(A,D)
checkpoint_dict_skip_on_merge=['cond_stage_model.transformer.text_model.embeddings.position_ids']
def to_half(tensor,enable):
	A=tensor
	if enable and A.dtype==torch.float:return A.half()
	return A
def read_metadata(primary_model_name,secondary_model_name,tertiary_model_name):
	A={}
	for C in[primary_model_name,secondary_model_name,tertiary_model_name]:
		B=sd_models.checkpoints_list.get(C,_A)
		if B is _A:continue
		A.update(B.metadata)
	return json.dumps(A,indent=4,ensure_ascii=_B)
def run_modelmerger(id_task,primary_model_name,secondary_model_name,tertiary_model_name,interp_method,multiplier,save_as_half,custom_name,checkpoint_format,config_source,bake_in_vae,discard_weights,save_metadata,add_merge_recipe,copy_metadata_fields,metadata_json):
	d='sd_merge_models';e='model';f=bake_in_vae;g=config_source;h=tertiary_model_name;i=secondary_model_name;j=primary_model_name;Q='sd_merge_recipe';R=save_metadata;S=discard_weights;T=custom_name;U=interp_method;P='cpu';M=save_as_half;K=multiplier;shared.state.begin(job='model-merge')
	def V(message):A=message;shared.state.textinfo=A;shared.state.end();return[*[gr.update()for A in range(4)],A]
	def m(theta0,theta1,alpha):A=alpha;return(1-A)*theta0+A*theta1
	def n(theta1,theta2):return theta1-theta2
	def o(theta0,theta1_2_diff,alpha):return theta0+alpha*theta1_2_diff
	def p():A=D.model_name;B=E.model_name;C=round(1-K,2);F=round(K,2);return f"{C}({A}) + {F}({B})"
	def q():A=D.model_name;B=E.model_name;C=H.model_name;F=round(K,2);return f"{A} + {F}({B} - {C})"
	def r():return D.model_name
	s={'Weighted sum':(p,_A,m),'Add difference':(q,n,o),'No interpolation':(r,_A,_A)};t,N,J=s[U];shared.state.job_count=(1 if N else 0)+(1 if J else 0)
	if not j:return V('Failed: Merging requires a primary model.')
	D=sd_models.checkpoints_list[j]
	if J and not i:return V('Failed: Merging requires a secondary model.')
	E=sd_models.checkpoints_list[i]if J else _A
	if N and not h:return V(f"Failed: Interpolation method ({U}) requires a tertiary model.")
	H=sd_models.checkpoints_list[h]if N else _A;W=_B;X=_B
	if J:shared.state.textinfo='Loading B';print(f"Loading {E.filename}...");C=sd_models.read_state_dict(E.filename,map_location=P)
	else:C=_A
	if N:
		shared.state.textinfo='Loading C';print(f"Loading {H.filename}...");Y=sd_models.read_state_dict(H.filename,map_location=P);shared.state.textinfo='Merging B and C';shared.state.sampling_steps=len(C.keys())
		for A in tqdm.tqdm(C.keys()):
			if A in checkpoint_dict_skip_on_merge:continue
			if e in A:
				if A in Y:u=Y.get(A,torch.zeros_like(C[A]));C[A]=N(C[A],u)
				else:C[A]=torch.zeros_like(C[A])
			shared.state.sampling_step+=1
		del Y;shared.state.nextjob()
	shared.state.textinfo=f"Loading {D.filename}...";print(f"Loading {D.filename}...");B=sd_models.read_state_dict(D.filename,map_location=P);print('Merging...');shared.state.textinfo='Merging A and B';shared.state.sampling_steps=len(B.keys())
	for A in tqdm.tqdm(B.keys()):
		if C and e in A and A in C:
			if A in checkpoint_dict_skip_on_merge:continue
			F=B[A];G=C[A]
			if F.shape!=G.shape and F.shape[0:1]+F.shape[2:]==G.shape[0:1]+G.shape[2:]:
				if F.shape[1]==4 and G.shape[1]==9:raise RuntimeError('When merging inpainting model with a normal one, A must be the inpainting model.')
				if F.shape[1]==4 and G.shape[1]==8:raise RuntimeError('When merging instruct-pix2pix model with a normal one, A must be the instruct-pix2pix model.')
				if F.shape[1]==8 and G.shape[1]==4:B[A][:,0:4,:,:]=J(F[:,0:4,:,:],G,K);X=True
				else:assert F.shape[1]==9 and G.shape[1]==4,f"Bad dimensions for merged layer {A}: A={F.shape}, B={G.shape}";B[A][:,0:4,:,:]=J(F[:,0:4,:,:],G,K);W=True
			else:B[A]=J(F,G,K)
			B[A]=to_half(B[A],M)
		shared.state.sampling_step+=1
	del C;Z=sd_vae.vae_dict.get(f,_A)
	if Z is not _A:
		print(f"Baking in VAE from {Z}");shared.state.textinfo='Baking in VAE';a=sd_vae.load_vae_dict(Z,map_location=P)
		for A in a.keys():
			k='first_stage_model.'+A
			if k in B:B[k]=to_half(a[A],M)
		del a
	if M and not J:
		for A in B.keys():B[A]=to_half(B[A],M)
	if S:
		v=re.compile(S)
		for A in list(B):
			if re.search(v,A):B.pop(A,_A)
	w=shared.cmd_opts.ckpt_dir or sd_models.model_path;O=t()if T==''else T;O+='.inpainting'if W else'';O+='.instruct-pix2pix'if X else'';O+='.'+checkpoint_format;L=os.path.join(w,O);shared.state.nextjob();shared.state.textinfo='Saving';print(f"Saving to {L}...");I={}
	if R and copy_metadata_fields:
		if D:I.update(D.metadata)
		if E:I.update(E.metadata)
		if H:I.update(H.metadata)
	if R:
		try:I.update(json.loads(metadata_json))
		except Exception as x:errors.display(x,'readin metadata from json')
		I['format']='pt'
	if R and add_merge_recipe:
		y={'type':'ourui','primary_model_hash':D.sha256,'secondary_model_hash':E.sha256 if E else _A,'tertiary_model_hash':H.sha256 if H else _A,'interp_method':U,'multiplier':K,'save_as_half':M,'custom_name':T,'config_source':g,'bake_in_vae':f,'discard_weights':S,'is_inpainting':W,'is_instruct_pix2pix':X};b={}
		def c(checkpoint_info):A=checkpoint_info;A.calculate_shorthash();b[A.sha256]={'name':A.name,'legacy_hash':A.hash,Q:A.metadata.get(Q,_A)};b.update(A.metadata.get(d,{}))
		c(D)
		if E:c(E)
		if H:c(H)
		I[Q]=json.dumps(y);I[d]=json.dumps(b)
	_,z=os.path.splitext(L)
	if z.lower()=='.safetensors':safetensors.torch.save_file(B,L,metadata=I if len(I)>0 else _A)
	else:torch.save(B,L)
	sd_models.list_models();l=next((A for A in sd_models.checkpoints_list.values()if A.name==O),_A)
	if l:l.calculate_shorthash()
	create_config(L,g,D,E,H);print(f"Checkpoint saved to {L}.");shared.state.textinfo='Checkpoint saved';shared.state.end();return[*[gr.Dropdown.update(choices=sd_models.checkpoint_tiles())for A in range(4)],'Checkpoint saved to '+L]