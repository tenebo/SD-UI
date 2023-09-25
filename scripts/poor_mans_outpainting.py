_C='down'
_B='right'
_A='left'
import math,modules.scripts as scripts,gradio as gr
from PIL import Image,ImageDraw
from modules import images,devices
from modules.processing import Processed,process_images
from modules.shared import opts,state
class Script(scripts.Script):
	def title(A):return"Poor man's outpainting"
	def show(A,is_img2img):return is_img2img
	def ui(A,is_img2img):
		B='fill'
		if not is_img2img:return
		C=gr.Slider(label='Pixels to expand',minimum=8,maximum=256,step=8,value=128,elem_id=A.elem_id('pixels'));D=gr.Slider(label='Mask blur',minimum=0,maximum=64,step=1,value=4,elem_id=A.elem_id('mask_blur'));E=gr.Radio(label='Masked content',choices=[B,'original','latent noise','latent nothing'],value=B,type='index',elem_id=A.elem_id('inpainting_fill'));F=gr.CheckboxGroup(label='Outpainting direction',choices=[_A,_B,'up',_C],value=[_A,_B,'up',_C],elem_id=A.elem_id('direction'));return[C,D,E,F]
	def run(t,p,pixels,mask_blur,inpainting_fill,direction):
		j='black';i='white';h='RGB';c=None;O=direction;G=pixels;F=mask_blur;P=c;S=c;p.mask_blur=F*2;p.inpainting_fill=inpainting_fill;p.inpaint_full_res=False;A=G if _A in O else 0;C=G if _B in O else 0;B=G if'up'in O else 0;D=G if _C in O else 0;H=p.init_images[0];T=math.ceil((H.width+A+C)/64)*64;U=math.ceil((H.height+B+D)/64)*64
		if A>0:A=A*(T-H.width)//(A+C)
		if C>0:C=T-H.width-A
		if B>0:B=B*(U-H.height)//(B+D)
		if D>0:D=U-H.height-B
		E=Image.new(h,(T,U));E.paste(H,(A,B));I=Image.new('L',(E.width,E.height),i);k=ImageDraw.Draw(I);k.rectangle((A+(F*2 if A>0 else 0),B+(F*2 if B>0 else 0),I.width-C-(F*2 if C>0 else 0),I.height-D-(F*2 if D>0 else 0)),fill=j);d=Image.new('L',(E.width,E.height),i);l=ImageDraw.Draw(d);l.rectangle((A+(F//2 if A>0 else 0),B+(F//2 if B>0 else 0),I.width-C-(F//2 if C>0 else 0),I.height-D-(F//2 if D>0 else 0)),fill=j);devices.torch_gc();K=images.split_grid(E,tile_w=p.width,tile_h=p.height,overlap=G);m=images.split_grid(I,tile_w=p.width,tile_h=p.height,overlap=G);n=images.split_grid(d,tile_w=p.width,tile_h=p.height,overlap=G);p.n_iter=1;p.batch_size=1;p.do_not_save_grid=True;p.do_not_save_samples=True;Q=[];e=[];f=[];V=[]
		for((L,W,X),(Y,Y,o),(Y,Y,q))in zip(K.tiles,m.tiles,n.tiles):
			for(M,r,s)in zip(X,o,q):
				N,Z=M[0:2]
				if N>=A and N+Z<=E.width-C and L>=B and L+W<=E.height-D:continue
				Q.append(M[2]);e.append(r[2]);f.append(s[2])
		a=len(Q);print(f"Poor man's outpainting will process a total of {len(Q)} images tiled as {len(K.tiles[0][2])}x{len(K.tiles)}.");state.job_count=a
		for R in range(a):
			p.init_images=[Q[R]];p.image_mask=e[R];p.latent_mask=f[R];state.job=f"Batch {R+1} out of {a}";J=process_images(p)
			if P is c:P=J.seed;S=J.info
			p.seed=J.seed+1;V+=J.images
		b=0
		for(L,W,X)in K.tiles:
			for M in X:
				N,Z=M[0:2]
				if N>=A and N+Z<=E.width-C and L>=B and L+W<=E.height-D:continue
				M[2]=V[b]if b<len(V)else Image.new(h,(p.width,p.height));b+=1
		g=images.combine_grid(K)
		if opts.samples_save:images.save_image(g,p.outpath_samples,'',P,p.prompt,opts.samples_format,info=S,p=p)
		J=Processed(p,[g],P,S);return J