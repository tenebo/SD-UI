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
		d='black';e='white';f='RGB';S=None;O=direction;G=pixels;F=mask_blur;P=S;T=S;p.mask_blur=F*2;p.inpainting_fill=inpainting_fill;p.inpaint_full_res=False;A=G if _A in O else 0;C=G if _B in O else 0;B=G if'up'in O else 0;D=G if _C in O else 0;H=p.init_images[0];U=math.ceil((H.width+A+C)/64)*64;V=math.ceil((H.height+B+D)/64)*64
		if A>0:A=A*(U-H.width)//(A+C)
		if C>0:C=U-H.width-A
		if B>0:B=B*(V-H.height)//(B+D)
		if D>0:D=V-H.height-B
		E=Image.new(f,(U,V));E.paste(H,(A,B));I=Image.new('L',(E.width,E.height),e);k=ImageDraw.Draw(I);k.rectangle((A+(F*2 if A>0 else 0),B+(F*2 if B>0 else 0),I.width-C-(F*2 if C>0 else 0),I.height-D-(F*2 if D>0 else 0)),fill=d);g=Image.new('L',(E.width,E.height),e);l=ImageDraw.Draw(g);l.rectangle((A+(F//2 if A>0 else 0),B+(F//2 if B>0 else 0),I.width-C-(F//2 if C>0 else 0),I.height-D-(F//2 if D>0 else 0)),fill=d);devices.torch_gc();K=images.split_grid(E,tile_w=p.width,tile_h=p.height,overlap=G);m=images.split_grid(I,tile_w=p.width,tile_h=p.height,overlap=G);n=images.split_grid(g,tile_w=p.width,tile_h=p.height,overlap=G);p.n_iter=1;p.batch_size=1;p.do_not_save_grid=True;p.do_not_save_samples=True;Q=[];h=[];i=[];W=[]
		for((L,X,Y),(Z,Z,o),(Z,Z,q))in zip(K.tiles,m.tiles,n.tiles):
			for(M,r,s)in zip(Y,o,q):
				N,a=M[0:2]
				if N>=A and N+a<=E.width-C and L>=B and L+X<=E.height-D:continue
				Q.append(M[2]);h.append(r[2]);i.append(s[2])
		b=len(Q);print(f"Poor man's outpainting will process a total of {len(Q)} images tiled as {len(K.tiles[0][2])}x{len(K.tiles)}.");state.job_count=b
		for R in range(b):
			p.init_images=[Q[R]];p.image_mask=h[R];p.latent_mask=i[R];state.job=f"Batch {R+1} out of {b}";J=process_images(p)
			if P is S:P=J.seed;T=J.info
			p.seed=J.seed+1;W+=J.images
		c=0
		for(L,X,Y)in K.tiles:
			for M in Y:
				N,a=M[0:2]
				if N>=A and N+a<=E.width-C and L>=B and L+X<=E.height-D:continue
				M[2]=W[c]if c<len(W)else Image.new(f,(p.width,p.height));c+=1
		j=images.combine_grid(K)
		if opts.samples_save:images.save_image(j,p.outpath_samples,'',P,p.prompt,opts.samples_format,info=T,p=p)
		J=Processed(p,[j],P,T);return J