_B=False
_A=None
import os
from PIL import Image,ImageOps
import math,tqdm
from modules import paths,shared,images,deepbooru
from modules.textual_inversion import autocrop
def preprocess(id_task,process_src,process_dst,process_width,process_height,preprocess_txt_action,process_keep_original_size,process_flip,process_split,process_caption,process_caption_deepbooru=_B,split_threshold=.5,overlap_ratio=.2,process_focal_crop=_B,process_focal_crop_face_weight=.9,process_focal_crop_entropy_weight=.15,process_focal_crop_edges_weight=.5,process_focal_crop_debug=_B,process_multicrop=_A,process_multicrop_mindim=_A,process_multicrop_maxdim=_A,process_multicrop_minarea=_A,process_multicrop_maxarea=_A,process_multicrop_objective=_A,process_multicrop_threshold=_A):
	A=process_caption_deepbooru;B=process_caption
	try:
		if B:shared.interrogator.load()
		if A:deepbooru.model.start()
		preprocess_work(process_src,process_dst,process_width,process_height,preprocess_txt_action,process_keep_original_size,process_flip,process_split,B,A,split_threshold,overlap_ratio,process_focal_crop,process_focal_crop_face_weight,process_focal_crop_entropy_weight,process_focal_crop_edges_weight,process_focal_crop_debug,process_multicrop,process_multicrop_mindim,process_multicrop_maxdim,process_multicrop_minarea,process_multicrop_maxarea,process_multicrop_objective,process_multicrop_threshold)
	finally:
		if B:shared.interrogator.send_blip_to_ram()
		if A:deepbooru.model.stop()
def listfiles(dirname):return os.listdir(dirname)
class PreprocessParams:src=_A;dstdir=_A;subindex=0;flip=_B;process_caption=_B;process_caption_deepbooru=_B;preprocess_txt_action=_A
def save_pic_with_caption(image,index,params,existing_caption=_A):
	E=image;C=existing_caption;B=params;A=''
	if B.process_caption:A+=shared.interrogator.generate_caption(E)
	if B.process_caption_deepbooru:
		if A:A+=', '
		A+=deepbooru.model.tag_multi(E)
	D=B.src;D=os.path.splitext(D)[0];D=os.path.basename(D);F=f"{index:05}-{B.subindex}-{D}";E.save(os.path.join(B.dstdir,f"{F}.png"))
	if B.preprocess_txt_action=='prepend'and C:A=f"{C} {A}"
	elif B.preprocess_txt_action=='append'and C:A=f"{A} {C}"
	elif B.preprocess_txt_action=='copy'and C:A=C
	A=A.strip()
	if A:
		with open(os.path.join(B.dstdir,f"{F}.txt"),'w',encoding='utf8')as G:G.write(A)
	B.subindex+=1
def save_pic(image,index,params,existing_caption=_A):
	B=existing_caption;C=index;D=image;A=params;save_pic_with_caption(D,C,A,existing_caption=B)
	if A.flip:save_pic_with_caption(ImageOps.mirror(D),C,A,existing_caption=B)
def split_pic(image,inverse_xy,width,height,overlap_ratio):
	G=overlap_ratio;H=height;I=width;F=inverse_xy;A=image
	if F:J,K=A.height,A.width;B,C=H,I
	else:J,K=A.width,A.height;B,C=I,H
	D=K*B//J
	if F:A=A.resize((D,B))
	else:A=A.resize((B,D))
	L=math.ceil((D-C*G)/(C*(1.-G)));N=(D-C)/(L-1)
	for O in range(L):
		E=int(N*O)
		if F:M=A.crop((E,0,E+C,B))
		else:M=A.crop((0,E,B,E+C))
		yield M
def center_crop(image,w,h):
	C=image;A,B=C.size
	if B/h<A/w:D=w*B/h;E=(A-D)/2,0,A-(A-D)/2,B
	else:F=h*A/w;E=0,(B-F)/2,A,B-(B-F)/2
	return C.resize((w,h),Image.Resampling.LANCZOS,E)
def multicrop_pic(image,mindim,maxdim,minarea,maxarea,objective,threshold):C=maxdim;D=mindim;A=image;F,G=A.size;E=lambda w,h:1-(lambda x:x if x<1 else 1/x)(F/G/(w/h));B=max(((A,B)for A in range(D,C+1,64)for B in range(D,C+1,64)if minarea<=A*B<=maxarea and E(A,B)<=threshold),key=lambda wh:(wh[0]*wh[1],-E(*wh))[::1 if objective=='Maximize area'else-1],default=_A);return B and center_crop(A,*B)
def preprocess_work(process_src,process_dst,process_width,process_height,preprocess_txt_action,process_keep_original_size,process_flip,process_split,process_caption,process_caption_deepbooru=_B,split_threshold=.5,overlap_ratio=.2,process_focal_crop=_B,process_focal_crop_face_weight=.9,process_focal_crop_entropy_weight=.3,process_focal_crop_edges_weight=.5,process_focal_crop_debug=_B,process_multicrop=_A,process_multicrop_mindim=_A,process_multicrop_maxdim=_A,process_multicrop_minarea=_A,process_multicrop_maxarea=_A,process_multicrop_objective=_A,process_multicrop_threshold=_A):
	I=True;J=overlap_ratio;K=split_threshold;E=process_width;F=process_height;L=os.path.abspath(process_src);M=os.path.abspath(process_dst);K=max(.0,min(1.,K));J=max(.0,min(.9,J));assert L!=M,'same directory specified as source and destination';os.makedirs(M,exist_ok=I);N=listfiles(L);shared.state.job='preprocess';shared.state.textinfo='Preprocessing...';shared.state.job_count=len(N);B=PreprocessParams();B.dstdir=M;B.flip=process_flip;B.process_caption=process_caption;B.process_caption_deepbooru=process_caption_deepbooru;B.preprocess_txt_action=preprocess_txt_action;P=tqdm.tqdm(N)
	for(C,V)in enumerate(P):
		B.subindex=0;H=os.path.join(L,V)
		try:A=Image.open(H);A=ImageOps.exif_transpose(A);A=A.convert('RGB')
		except Exception:continue
		Q=f"Preprocessing [Image {C}/{len(N)}]";P.set_description(Q);shared.state.textinfo=Q;B.src=H;D=_A;R=f"{os.path.splitext(H)[0]}.txt"
		if os.path.exists(R):
			with open(R,'r',encoding='utf8')as W:D=W.read()
		if shared.state.interrupted:break
		if A.height>A.width:O=A.width*F/(A.height*E);S=_B
		else:O=A.height*E/(A.width*F);S=I
		G=I
		if process_split and O<1. and O<=K:
			for X in split_pic(A,S,E,F,J):save_pic(X,C,B,existing_caption=D)
			G=_B
		if process_focal_crop and A.height!=A.width:
			T=_A
			try:T=autocrop.download_and_cache_models(os.path.join(paths.models_path,'opencv'))
			except Exception as Y:print('Unable to load face detection model for auto crop selection. Falling back to lower quality haar method.',Y)
			Z=autocrop.Settings(crop_width=E,crop_height=F,face_points_weight=process_focal_crop_face_weight,entropy_points_weight=process_focal_crop_entropy_weight,corner_points_weight=process_focal_crop_edges_weight,annotate_image=process_focal_crop_debug,dnn_model_path=T)
			for a in autocrop.crop_image(A,Z):save_pic(a,C,B,existing_caption=D)
			G=_B
		if process_multicrop:
			U=multicrop_pic(A,process_multicrop_mindim,process_multicrop_maxdim,process_multicrop_minarea,process_multicrop_maxarea,process_multicrop_objective,process_multicrop_threshold)
			if U is not _A:save_pic(U,C,B,existing_caption=D)
			else:print(f"skipped {A.width}x{A.height} image {H} (can't find suitable size within error threshold)")
			G=_B
		if process_keep_original_size:save_pic(A,C,B,existing_caption=D);G=_B
		if G:A=images.resize_image(1,A,E,F);save_pic(A,C,B,existing_caption=D)
		shared.state.nextjob()