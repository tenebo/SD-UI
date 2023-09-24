from __future__ import annotations
_K='job_timestamp'
_J='sampler'
_I='exif'
_H='utf8'
_G='RGBA'
_F='parameters'
_E='None'
_D=True
_C=False
_B='RGB'
_A=None
import datetime,pytz,io,math,os
from collections import namedtuple
import re,numpy as np,piexif,piexif.helper
from PIL import Image,ImageFont,ImageDraw,ImageColor,PngImagePlugin
import string,json,hashlib
from modules import sd_samplers,shared,script_callbacks,errors
from modules.paths_internal import roboto_ttf_file
from modules.shared import opts
LANCZOS=Image.Resampling.LANCZOS if hasattr(Image,'Resampling')else Image.LANCZOS
def get_font(fontsize):
	A=fontsize
	try:return ImageFont.truetype(opts.font or roboto_ttf_file,A)
	except Exception:return ImageFont.truetype(roboto_ttf_file,A)
def image_grid(imgs,batch_size=1,rows=_A):
	B=imgs;A=rows
	if A is _A:
		if opts.n_rows>0:A=opts.n_rows
		elif opts.n_rows==0:A=batch_size
		elif opts.grid_prevent_empty_spots:
			A=math.floor(math.sqrt(len(B)))
			while len(B)%A!=0:A-=1
		else:A=math.sqrt(len(B));A=round(A)
	if A>len(B):A=len(B)
	H=math.ceil(len(B)/A);C=script_callbacks.ImageGridLoopParams(B,H,A);script_callbacks.image_grid_callback(C);D,E=B[0].size;F=Image.new(_B,size=(C.cols*D,C.rows*E),color='black')
	for(G,I)in enumerate(C.imgs):F.paste(I,box=(G%C.cols*D,G//C.cols*E))
	return F
Grid=namedtuple('Grid',['tiles','tile_w','tile_h','image_w','image_h','overlap'])
def split_grid(image,tile_w=512,tile_h=512,overlap=64):
	H=image;C=overlap;B=tile_h;A=tile_w;D=H.width;E=H.height;M=A-C;N=B-C;I=math.ceil((D-C)/M);J=math.ceil((E-C)/N);O=(D-A)/(I-1)if I>1 else 0;P=(E-B)/(J-1)if J>1 else 0;K=Grid([],A,B,D,E,C)
	for Q in range(J):
		L=[];F=int(Q*P)
		if F+B>=E:F=E-B
		for R in range(I):
			G=int(R*O)
			if G+A>=D:G=D-A
			S=H.crop((G,F,G+A,F+B));L.append([G,A,S])
		K.tiles.append([F,B,L])
	return K
def combine_grid(grid):
	A=grid
	def H(r):r=r*255/A.overlap;r=r.astype(np.uint8);return Image.fromarray(r,'L')
	I=H(np.arange(A.overlap,dtype=np.float32).reshape((1,A.overlap)).repeat(A.tile_h,axis=0));J=H(np.arange(A.overlap,dtype=np.float32).reshape((A.overlap,1)).repeat(A.image_w,axis=1));C=Image.new(_B,(A.image_w,A.image_h))
	for(E,D,K)in A.tiles:
		B=Image.new(_B,(A.image_w,D))
		for(F,L,G)in K:
			if F==0:B.paste(G,(0,0));continue
			B.paste(G.crop((0,0,A.overlap,D)),(F,0),mask=I);B.paste(G.crop((A.overlap,0,L,D)),(F+A.overlap,0))
		if E==0:C.paste(B,(0,0));continue
		C.paste(B.crop((0,0,B.width,A.overlap)),(0,E),mask=J);C.paste(B.crop((0,A.overlap,B.width,D)),(0,E+A.overlap))
	return C
class GridAnnotation:
	def __init__(A,text='',is_active=_D):A.text=text;A.is_active=is_active;A.size=_A
def draw_grid_annotations(im,width,height,hor_texts,ver_texts,margin=0):
	J=im;G=margin;F=hor_texts;C=ver_texts;B=height;A=width;c=ImageColor.getcolor(opts.grid_text_active_color,_B);W=ImageColor.getcolor(opts.grid_text_inactive_color,_B);X=ImageColor.getcolor(opts.grid_background_color,_B)
	def d(drawing,text,font,line_length):
		A=['']
		for B in text.split():
			C=f"{A[-1]} {B}".strip()
			if drawing.textlength(C,font=font)<=line_length:A[-1]=C
			else:A.append(B)
		return A
	def Y(drawing,draw_x,draw_y,lines,initial_fnt,initial_fontsize):
		D=draw_x;C=drawing;B=draw_y
		for A in lines:
			E=initial_fnt;F=initial_fontsize
			while C.multiline_textsize(A.text,font=E)[0]>A.allowed_width and F>0:F-=1;E=get_font(F)
			C.multiline_text((D,B+A.size[1]/2),A.text,font=E,fill=c if A.is_active else W,anchor='mm',align='center')
			if not A.is_active:C.line((D-A.size[0]//2,B+A.size[1]//2,D+A.size[0]//2,B+A.size[1]//2),fill=W,width=4)
			B+=A.size[1]+H
	N=(A+B)//25;H=N//2;O=get_font(N);K=0 if sum([sum([len(A.text)for A in A])for A in C])==0 else A*3//4;L=J.width//A;M=J.height//B;assert L==len(F),f"bad number of horizontal texts: {len(F)}; must be {L}";assert M==len(C),f"bad number of vertical texts: {len(C)}; must be {M}";e=Image.new(_B,(1,1),X);Z=ImageDraw.Draw(e)
	for(P,a)in zip(F+C,[A]*len(F)+[K]*len(C)):
		f=[]+P;P.clear()
		for I in f:g=d(Z,I.text,O,a);P+=[GridAnnotation(A,I.is_active)for A in g]
		for I in P:Q=Z.multiline_textbbox((0,0),I.text,font=O);I.size=Q[2]-Q[0],Q[3]-Q[1];I.allowed_width=a
	S=[sum([A.size[1]+H for A in A])-H for A in F];h=[sum([A.size[1]+H for A in A])-H*len(A)for A in C];R=0 if sum(S)==0 else max(S)+H*2;T=Image.new(_B,(J.width+K+G*(L-1),J.height+R+G*(M-1)),X)
	for D in range(M):
		for E in range(L):i=J.crop((A*E,B*D,A*(E+1),B*(D+1)));T.paste(i,(K+(A+G)*E,R+(B+G)*D))
	b=ImageDraw.Draw(T)
	for E in range(L):U=K+(A+G)*E+A/2;V=R/2-S[E]/2;Y(b,U,V,F[E],O,N)
	for D in range(M):U=K/2;V=R+(B+G)*D+B/2-h[D]/2;Y(b,U,V,C[D],O,N)
	return T
def draw_prompt_matrix(im,width,height,all_prompts,margin=0):A=all_prompts[1:];B=math.ceil(len(A)/2);C=A[:B];D=A[B:];E=[[GridAnnotation(C,is_active=A&1<<B!=0)for(B,C)in enumerate(C)]for A in range(1<<len(C))];F=[[GridAnnotation(C,is_active=A&1<<B!=0)for(B,C)in enumerate(D)]for A in range(1<<len(D))];return draw_grid_annotations(im,width,height,E,F,margin)
def resize_image(resize_mode,im,width,height,upscaler_name=_A):
	'\n    Resizes an image with the specified resize_mode, width, and height.\n\n    Args:\n        resize_mode: The mode to use when resizing the image.\n            0: Resize the image to the specified width and height.\n            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.\n            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.\n        im: The image to resize.\n        width: The width to resize the image to.\n        height: The height to resize the image to.\n        upscaler_name: The name of the upscaler to use. If not provided, defaults to opts.upscaler_for_img2img.\n    ';N=resize_mode;J=upscaler_name;C=im;B=height;A=width;J=J or opts.upscaler_for_img2img
	def M(im,w,h):
		A=im
		if J is _A or J==_E or A.mode=='L':return A.resize((w,h),resample=LANCZOS)
		C=max(w/A.width,h/A.height)
		if C>1.:
			D=[A for A in shared.sd_upscalers if A.name==J]
			if len(D)==0:B=shared.sd_upscalers[0];print(f"could not find upscaler named {J or'<empty string>'}, using {B.name} as a fallback")
			else:B=D[0]
			A=B.scaler.upscale(A,C,B.data_path)
		if A.width!=w or A.height!=h:A=A.resize((w,h),resample=LANCZOS)
		return A
	if N==0:E=M(C,A,B)
	elif N==1:F=A/B;G=C.width/C.height;H=A if F>G else C.width*B//C.height;I=B if F<=G else C.height*A//C.width;D=M(C,H,I);E=Image.new(_B,(A,B));E.paste(D,box=(A//2-H//2,B//2-I//2))
	else:
		F=A/B;G=C.width/C.height;H=A if F<G else C.width*B//C.height;I=B if F>=G else C.height*A//C.width;D=M(C,H,I);E=Image.new(_B,(A,B));E.paste(D,box=(A//2-H//2,B//2-I//2))
		if F<G:
			K=B//2-I//2
			if K>0:E.paste(D.resize((A,K),box=(0,0,A,0)),box=(0,0));E.paste(D.resize((A,K),box=(0,D.height,A,D.height)),box=(0,K+I))
		elif F>G:
			L=A//2-H//2
			if L>0:E.paste(D.resize((L,B),box=(0,0,0,B)),box=(0,0));E.paste(D.resize((L,B),box=(D.width,0,D.width,B)),box=(L+H,0))
	return E
invalid_filename_chars='<>:"/\\|?*\n\r\t'
invalid_filename_prefix=' '
invalid_filename_postfix=' .'
re_nonletters=re.compile('[\\s'+string.punctuation+']+')
re_pattern=re.compile('(.*?)(?:\\[([^\\[\\]]+)\\]|$)')
re_pattern_arg=re.compile('(.*)<([^>]*)>$')
max_filename_part_length=128
NOTHING_AND_SKIP_PREVIOUS_TEXT=object()
def sanitize_filename_part(text,replace_spaces=_D):
	A=text
	if A is _A:return
	if replace_spaces:A=A.replace(' ','_')
	A=A.translate({ord(A):'_'for A in invalid_filename_chars});A=A.lstrip(invalid_filename_prefix)[:max_filename_part_length];A=A.rstrip(invalid_filename_postfix);return A
class FilenameGenerator:
	replacements={'seed':lambda self:self.seed if self.seed is not _A else'','seed_first':lambda self:self.seed if self.p.batch_size==1 else self.p.all_seeds[0],'seed_last':lambda self:NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.batch_size==1 else self.p.all_seeds[-1],'steps':lambda self:self.p and self.p.steps,'cfg':lambda self:self.p and self.p.cfg_scale,'width':lambda self:self.image.width,'height':lambda self:self.image.height,'styles':lambda self:self.p and sanitize_filename_part(', '.join([A for A in self.p.styles if not A==_E])or _E,replace_spaces=_C),_J:lambda self:self.p and sanitize_filename_part(self.p.sampler_name,replace_spaces=_C),'model_hash':lambda self:getattr(self.p,'sd_model_hash',shared.sd_model.sd_model_hash),'model_name':lambda self:sanitize_filename_part(shared.sd_model.sd_checkpoint_info.name_for_extra,replace_spaces=_C),'date':lambda self:datetime.datetime.now().strftime('%Y-%m-%d'),'datetime':lambda self,*A:self.datetime(*A),_K:lambda self:getattr(self.p,_K,shared.state.job_timestamp),'prompt_hash':lambda self,*A:self.string_hash(self.prompt,*A),'negative_prompt_hash':lambda self,*A:self.string_hash(self.p.negative_prompt,*A),'full_prompt_hash':lambda self,*A:self.string_hash(f"{self.p.prompt} {self.p.negative_prompt}",*A),'prompt':lambda self:sanitize_filename_part(self.prompt),'prompt_no_styles':lambda self:self.prompt_no_style(),'prompt_spaces':lambda self:sanitize_filename_part(self.prompt,replace_spaces=_C),'prompt_words':lambda self:self.prompt_words(),'batch_number':lambda self:NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.batch_size==1 or self.zip else self.p.batch_index+1,'batch_size':lambda self:self.p.batch_size,'generation_number':lambda self:NOTHING_AND_SKIP_PREVIOUS_TEXT if self.p.n_iter==1 and self.p.batch_size==1 or self.zip else self.p.iteration*self.p.batch_size+self.p.batch_index+1,'hasprompt':lambda self,*A:self.hasprompt(*A),'clip_skip':lambda self:opts.data['CLIP_stop_at_last_layers'],'denoising':lambda self:self.p.denoising_strength if self.p and self.p.denoising_strength else NOTHING_AND_SKIP_PREVIOUS_TEXT,'user':lambda self:self.p.user,'vae_filename':lambda self:self.get_vae_filename(),'none':lambda self:'','image_hash':lambda self,*A:self.image_hash(*A)};default_time_format='%Y%m%d%H%M%S'
	def __init__(A,p,seed,prompt,image,zip=_C):A.p=p;A.seed=seed;A.prompt=prompt;A.image=image;A.zip=zip
	def get_vae_filename(D):
		'Get the name of the VAE file.';import modules.sd_vae as B
		if B.loaded_vae_file is _A:return'NoneType'
		C=os.path.basename(B.loaded_vae_file);A=C.split('.')
		if len(A)>1 and A[0]=='':return A[1]
		else:return A[0]
	def hasprompt(B,*G):
		H=B.prompt.lower()
		if B.p is _A or B.prompt is _A:return
		A=''
		for D in G:
			if D!='':
				C=D.split('|');E=C[0].lower();F=C[1]if len(C)>1 else''
				if H.find(E)>=0:A=f"{A}{E}"
				else:A=A if F==''else f"{A}{F}"
		return sanitize_filename_part(A)
	def prompt_no_style(B):
		D=','
		if B.p is _A or B.prompt is _A:return
		A=B.prompt
		for C in shared.prompt_styles.get_style_prompts(B.p.styles):
			if C:
				for E in C.split('{prompt}'):A=A.replace(E,'').replace(', ,',D).strip().strip(D)
				A=A.replace(C,'').strip().strip(D).strip()
		return sanitize_filename_part(A,replace_spaces=_C)
	def prompt_words(B):
		A=[A for A in re_nonletters.split(B.prompt or'')if A]
		if len(A)==0:A=['empty']
		return sanitize_filename_part(' '.join(A[0:opts.directories_max_prompt_words]),replace_spaces=_C)
	def datetime(B,*A):
		F=datetime.datetime.now();G=A[0]if A and A[0]!=''else B.default_time_format
		try:C=pytz.timezone(A[1])if len(A)>1 else _A
		except pytz.exceptions.UnknownTimeZoneError:C=_A
		D=F.astimezone(C)
		try:E=D.strftime(G)
		except(ValueError,TypeError):E=D.strftime(B.default_time_format)
		return sanitize_filename_part(E,replace_spaces=_C)
	def image_hash(B,*A):C=int(A[0])if A and A[0]!=''else _A;return hashlib.sha256(B.image.tobytes()).hexdigest()[0:C]
	def string_hash(C,text,*A):B=int(A[0])if A and A[0]!=''else 8;return hashlib.sha256(text.encode()).hexdigest()[0:B]
	def apply(F,x):
		B=''
		for C in re_pattern.finditer(x):
			E,A=C.groups()
			if A is _A:B+=E;continue
			G=[]
			while _D:
				C=re_pattern_arg.match(A)
				if C is _A:break
				A,I=C.groups();G.insert(0,I)
			H=F.replacements.get(A.lower())
			if H is not _A:
				try:D=H(F,*G)
				except Exception:D=_A;errors.report(f"Error adding [{A}] to filename",exc_info=_D)
				if D==NOTHING_AND_SKIP_PREVIOUS_TEXT:continue
				elif D is not _A:B+=E+str(D);continue
			B+=f"{E}[{A}]"
		return B
def get_next_sequence_number(path,basename):
	'\n    Determines and returns the next sequence number to use when saving an image in the specified directory.\n\n    The sequence starts at 0.\n    ';A=basename;B=-1
	if A!='':A=f"{A}-"
	D=len(A)
	for C in os.listdir(path):
		if C.startswith(A):
			E=os.path.splitext(C[D:])[0].split('-')
			try:B=max(int(E[0]),B)
			except ValueError:pass
	return B+1
def save_image_with_geninfo(image,geninfo,filename,extension=_A,existing_pnginfo=_A,pnginfo_section_name=_F):
	"\n    Saves image to filename, including geninfo as text information for generation info.\n    For PNG images, geninfo is added to existing pnginfo dictionary using the pnginfo_section_name argument as key.\n    For JPG images, there's no dictionary and geninfo just replaces the EXIF description.\n    ";H='.webp';E=geninfo;D=existing_pnginfo;C=filename;B=extension;A=image
	if B is _A:B=os.path.splitext(C)[1]
	F=Image.registered_extensions()[B]
	if B.lower()=='.png':
		D=D or{}
		if opts.enable_pnginfo:D[pnginfo_section_name]=E
		if opts.enable_pnginfo:
			G=PngImagePlugin.PngInfo()
			for(I,J)in(D or{}).items():G.add_text(I,str(J))
		else:G=_A
		A.save(C,format=F,quality=opts.jpeg_quality,pnginfo=G)
	elif B.lower()in('.jpg','.jpeg',H):
		if A.mode==_G:A=A.convert(_B)
		elif A.mode=='I;16':A=A.point(lambda p:p*.0038910505836576).convert(_B if B.lower()==H else'L')
		A.save(C,format=F,quality=opts.jpeg_quality,lossless=opts.webp_lossless)
		if opts.enable_pnginfo and E is not _A:K=piexif.dump({'Exif':{piexif.ExifIFD.UserComment:piexif.helper.UserComment.dump(E or'',encoding='unicode')}});piexif.insert(K,C)
	else:A.save(C,format=F,quality=opts.jpeg_quality)
def save_image(image,path,basename,seed=_A,prompt=_A,extension='png',info=_A,short_filename=_C,no_prompt=_C,grid=_C,pnginfo_section_name=_F,p=_A,existing_info=_A,forced_filename=_A,suffix='',save_to_dirs=_A):
	"Save an image.\n\n    Args:\n        image (`PIL.Image`):\n            The image to be saved.\n        path (`str`):\n            The directory to save the image. Note, the option `save_to_dirs` will make the image to be saved into a sub directory.\n        basename (`str`):\n            The base filename which will be applied to `filename pattern`.\n        seed, prompt, short_filename,\n        extension (`str`):\n            Image file extension, default is `png`.\n        pngsectionname (`str`):\n            Specify the name of the section which `info` will be saved in.\n        info (`str` or `PngImagePlugin.iTXt`):\n            PNG info chunks.\n        existing_info (`dict`):\n            Additional PNG info. `existing_info == {pngsectionname: info, ...}`\n        no_prompt:\n            TODO I don't know its meaning.\n        p (`StandardDemoProcessing`)\n        forced_filename (`str`):\n            If specified, `basename` and filename pattern will be ignored.\n        save_to_dirs (bool):\n            If true, the image will be saved into a subdirectory of `path`.\n\n    Returns: (fullfn, txt_fullfn)\n        fullfn (`str`):\n            The full path of the saved imaged.\n        txt_fullfn (`str` or None):\n            If a text file is saved for this image, this will be its full path. Otherwise None.\n    ";O=forced_filename;L=save_to_dirs;K=pnginfo_section_name;J=basename;G=info;F=path;B=extension;A=image;P=FilenameGenerator(p,seed,prompt,A)
	if(A.height>65535 or A.width>65535)and B.lower()in('jpg','jpeg')or(A.height>16383 or A.width>16383)and B.lower()=='webp':print('Image dimensions too large; saving as PNG');B='.png'
	if L is _A:L=grid and opts.grid_save_to_dirs or not grid and opts.save_to_dirs and not no_prompt
	if L:V=P.apply(opts.directories_filename_pattern or'[prompt_words]').lstrip(' ').rstrip('\\ /');F=os.path.join(F,V)
	os.makedirs(F,exist_ok=_D)
	if O is _A:
		if short_filename or seed is _A:C=''
		elif opts.save_to_dirs:C=opts.samples_filename_pattern or'[seed]'
		else:C=opts.samples_filename_pattern or'[seed]-[prompt_spaces]'
		C=P.apply(C)+suffix;Q=opts.save_images_add_number or C==''
		if C!=''and Q:C=f"-{C}"
		if Q:
			R=get_next_sequence_number(F,J);D=_A
			for S in range(500):
				W=f"{R+S:05}"if J==''else f"{J}-{R+S:04}";D=os.path.join(F,f"{W}{C}.{B}")
				if not os.path.exists(D):break
		else:D=os.path.join(F,f"{C}.{B}")
	else:D=os.path.join(F,f"{O}.{B}")
	T=existing_info or{}
	if G is not _A:T[K]=G
	E=script_callbacks.ImageSaveParams(A,p,D,T);script_callbacks.before_image_saved_callback(E);A=E.image;D=E.filename;G=E.pnginfo.get(K,_A)
	def U(image_to_save,filename_without_extension,extension):'\n        save image with .tmp extension to avoid race condition when another process detects new image in the directory\n        ';B=extension;A=filename_without_extension;C=f"{A}.tmp";save_image_with_geninfo(image_to_save,G,C,B,existing_pnginfo=E.pnginfo,pnginfo_section_name=K);os.replace(C,A+B)
	H,B=os.path.splitext(E.filename)
	if hasattr(os,'statvfs'):X=os.statvfs(F).f_namemax;H=H[:X-max(4,len(B))];E.filename=H+B;D=E.filename
	U(A,H,B);A.already_saved_as=D;M=A.width>opts.target_side_length or A.height>opts.target_side_length
	if opts.export_for_4chan and(M or os.stat(D).st_size>opts.img_downscale_threshold*1024*1024):
		Y=A.width/A.height;I=_A
		if M and Y>1:I=round(opts.target_side_length),round(A.height*opts.target_side_length/A.width)
		elif M:I=round(A.width*opts.target_side_length/A.height),round(opts.target_side_length)
		if I is not _A:
			try:A=A.resize(I,LANCZOS)
			except Exception:A=A.resize(I)
		try:U(A,H,'.jpg')
		except Exception as Z:errors.display(Z,'saving image as downscaled JPG')
	if opts.save_txt and G is not _A:
		N=f"{H}.txt"
		with open(N,'w',encoding=_H)as a:a.write(f"{G}\n")
	else:N=_A
	script_callbacks.image_saved_callback(E);return D,N
IGNORED_INFO_KEYS={'jfif','jfif_version','jfif_unit','jfif_density','dpi',_I,'loop','background','timestamp','duration','progressive','progression','icc_profile','chromaticity','photoshop'}
def read_info_from_image(image):
	D=image;A=(D.info or{}).copy();E=A.pop(_F,_A)
	if _I in A:
		F=piexif.load(A[_I]);B=(F or{}).get('Exif',{}).get(piexif.ExifIFD.UserComment,b'')
		try:B=piexif.helper.UserComment.load(B)
		except ValueError:B=B.decode(_H,errors='ignore')
		if B:A['exif comment']=B;E=B
	for G in IGNORED_INFO_KEYS:A.pop(G,_A)
	if A.get('Software',_A)=='NovelAI':
		try:C=json.loads(A['Comment']);H=sd_samplers.samplers_map.get(C[_J],'Euler a');E=f"{A['Description']}\nNegative prompt: {C['uc']}\nSteps: {C['steps']}, Sampler: {H}, CFG scale: {C['scale']}, Seed: {C['seed']}, Size: {D.width}x{D.height}, Clip skip: 2, ENSD: 31337"
		except Exception:errors.report('Error parsing NovelAI image generation parameters',exc_info=_D)
	return E,A
def image_data(data):
	import gradio as B
	try:C=Image.open(io.BytesIO(data));D,E=read_info_from_image(C);return D,_A
	except Exception:pass
	try:A=data.decode(_H);assert len(A)<10000;return A,_A
	except Exception:pass
	return B.update(),_A
def flatten(img,bgcolor):
	'replaces transparency with bgcolor (example: "#ffffff"), returning an RGB mode image with no transparency';A=img
	if A.mode==_G:B=Image.new(_G,A.size,bgcolor);B.paste(A,mask=A);A=B
	return A.convert(_B)