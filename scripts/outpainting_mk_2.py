_D='down'
_C='right'
_B='left'
_A=1.
import math,numpy as np,skimage,modules.scripts as scripts,gradio as gr
from PIL import Image,ImageDraw
from modules import images
from modules.processing import Processed,process_images
from modules.shared import opts,state
def get_matched_noise(_np_src_image,np_mask_rgb,noise_q=1,color_variation=.05):
	L=color_variation;F='ortho';D=np_mask_rgb;B=_np_src_image
	def G(data):
		A=data
		if A.ndim>2:
			B=np.zeros((A.shape[0],A.shape[1],A.shape[2]),dtype=np.complex128)
			for C in range(A.shape[2]):D=A[:,:,C];B[:,:,C]=np.fft.fft2(np.fft.fftshift(D),norm=F);B[:,:,C]=np.fft.ifftshift(B[:,:,C])
		else:B=np.zeros((A.shape[0],A.shape[1]),dtype=np.complex128);B[:,:]=np.fft.fft2(np.fft.fftshift(A),norm=F);B[:,:]=np.fft.ifftshift(B[:,:])
		return B
	def M(data):
		A=data
		if A.ndim>2:
			B=np.zeros((A.shape[0],A.shape[1],A.shape[2]),dtype=np.complex128)
			for C in range(A.shape[2]):D=A[:,:,C];B[:,:,C]=np.fft.ifft2(np.fft.fftshift(D),norm=F);B[:,:,C]=np.fft.ifftshift(B[:,:,C])
		else:B=np.zeros((A.shape[0],A.shape[1]),dtype=np.complex128);B[:,:]=np.fft.ifft2(np.fft.fftshift(A),norm=F);B[:,:]=np.fft.ifftshift(B[:,:])
		return B
	def U(width,height,std=3.14,mode=0):
		B=height;A=width;G=float(A/min(A,B));H=float(B/min(A,B));C=np.zeros((A,B));E=(np.arange(A)/A*2.-_A)*G
		for D in range(B):
			F=(D/B*2.-_A)*H
			if mode==0:C[:,D]=np.exp(-(E**2+F**2)*std)
			else:C[:,D]=(1/((E**2+_A)*(F**2+_A)))**(std/3.14)
		return C
	def V(np_mask_grey,hardness=_A):
		B=hardness;A=np_mask_grey;C=np.zeros((A.shape[0],A.shape[1],3))
		if B!=_A:D=A[:]**B
		else:D=A[:]
		for E in range(3):C[:,:,E]=D[:]
		return C
	N=B.shape[0];O=B.shape[1];H=B.shape[2];B[:]*(_A-D);I=np.sum(D,axis=2)/3.;P=I>1e-06;W=I<.001;E=B*(_A-V(I));E/=np.max(E);E+=np.average(B)*D;Q=G(E);R=np.absolute(Q);X=Q/R;Y=np.random.default_rng(0);Z=U(N,O,mode=1);C=Y.random((N,O,H));a=np.sum(C,axis=2)/3.;C*=L
	for J in range(H):C[:,:,J]+=(_A-L)*a
	S=G(C)
	for J in range(H):S[:,:,J]*=Z
	C=np.real(M(S));K=G(C);K[:,:,:]=np.absolute(K[:,:,:])**2*R**noise_q*X;T=.0;b=B[:]*(T+_A)-T*2.;A=np.real(M(K));A-=np.min(A);A/=np.max(A);A[P,:]=skimage.exposure.match_histograms(A[P,:]**_A,b[W,:],channel_axis=1);A=B[:]*(_A-D)+A*D;c=A[:];return np.clip(c,.0,_A)
class Script(scripts.Script):
	def title(A):return'Outpainting mk2'
	def show(A,is_img2img):return is_img2img
	def ui(A,is_img2img):
		if not is_img2img:return
		B=gr.HTML('<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>');C=gr.Slider(label='Pixels to expand',minimum=8,maximum=256,step=8,value=128,elem_id=A.elem_id('pixels'));D=gr.Slider(label='Mask blur',minimum=0,maximum=64,step=1,value=8,elem_id=A.elem_id('mask_blur'));E=gr.CheckboxGroup(label='Outpainting direction',choices=[_B,_C,'up',_D],value=[_B,_C,'up',_D],elem_id=A.elem_id('direction'));F=gr.Slider(label='Fall-off exponent (lower=higher detail)',minimum=.0,maximum=4.,step=.01,value=_A,elem_id=A.elem_id('noise_q'));G=gr.Slider(label='Color variation',minimum=.0,maximum=_A,step=.01,value=.05,elem_id=A.elem_id('color_variation'));return[B,C,D,E,F,G]
	def run(b,p,_,pixels,mask_blur,direction,noise_q,color_variation):
		T=mask_blur;S=None;P=direction;O=pixels;N=False;G=True;K=[S,S];h=p.width;i=p.height;p.inpaint_full_res=N;p.inpainting_fill=1;p.do_not_save_samples=G;p.do_not_save_grid=G;A=O if _B in P else 0;D=O if _C in P else 0;B=O if'up'in P else 0;E=O if _D in P else 0
		if A>0 or D>0:H=T
		else:H=0
		if B>0 or E>0:I=T
		else:I=0
		p.mask_blur_x=H*4;p.mask_blur_y=I*4;F=p.init_images[0];U=math.ceil((F.width+A+D)/64)*64;V=math.ceil((F.height+B+E)/64)*64
		if A>0:A=A*(U-F.width)//(A+D)
		if D>0:D=U-F.width-A
		if B>0:B=B*(V-F.height)//(B+E)
		if E>0:E=V-F.height-B
		def Q(init,count,expand_pixels,is_left=N,is_right=N,is_top=N,is_bottom=N):
			g='black';f='white';Z=count;Q=is_bottom;P=is_right;O='RGB';J=init;F=is_top;E=is_left;C=expand_pixels;R=E or P;T=F or Q;U=C if R else 0;V=C if T else 0;a=[];B=[]
			for A in range(Z):M=J[A].width+U;N=J[A].height+V;b=math.ceil(M/64)*64;c=math.ceil(N/64)*64;G=Image.new(O,(b,c));G.paste(J[A],(U if E else 0,V if F else 0));D=Image.new(O,(b,c),f);W=ImageDraw.Draw(D);W.rectangle((C+H if E else 0,C+I if F else 0,D.width-C-H if P else M,D.height-C-I if Q else N),fill=g);j=(np.asarray(G)/255.).astype(np.float64);k=(np.asarray(D)/255.).astype(np.float64);l=get_matched_noise(j,k,noise_q,color_variation);B.append(Image.fromarray(np.clip(l*255.,.0,255.).astype(np.uint8),mode=O));X=min(h,J[A].width+U)if R else G.width;Y=min(i,J[A].height+V)if T else G.height;p.width=X if R else G.width;p.height=Y if T else G.height;d=0 if E else B[A].width-X,0 if F else B[A].height-Y,X if E else B[A].width,Y if F else B[A].height;D=D.crop(d);p.image_mask=D;m=B[A].crop(d);a.append(m)
			p.init_images=a;e=Image.new(O,(p.width,p.height),f);W=ImageDraw.Draw(e);W.rectangle((C+H*2 if E else 0,C+I*2 if F else 0,D.width-C-H*2 if P else M,D.height-C-I*2 if Q else N),fill=g);p.latent_mask=e;L=process_images(p)
			if K[0]is S:K[0]=L.seed;K[1]=L.info
			for A in range(Z):B[A].paste(L.images[A],(0 if E else B[A].width-L.images[A].width,0 if F else B[A].height-L.images[A].height));B[A]=B[A].crop((0,0,M,N))
			return B
		R=p.n_iter;L=p.batch_size;p.n_iter=1;state.job_count=R*((1 if A>0 else 0)+(1 if D>0 else 0)+(1 if B>0 else 0)+(1 if E>0 else 0));J=[]
		for Z in range(R):
			C=[F]*L;state.job=f"Batch {Z+1} out of {R}"
			if A>0:C=Q(C,L,A,is_left=G)
			if D>0:C=Q(C,L,D,is_right=G)
			if B>0:C=Q(C,L,B,is_top=G)
			if E>0:C=Q(C,L,E,is_bottom=G)
			J+=C
		W=J;X=images.image_grid(J);Y=len(J)<2 and opts.grid_only_if_multiple
		if opts.return_grid and not Y:W=[X]+J
		M=Processed(p,W,K[0],K[1])
		if opts.samples_save:
			for a in J:images.save_image(a,p.outpath_samples,'',M.seed,p.prompt,opts.samples_format,info=M.info,p=p)
		if opts.grid_save and not Y:images.save_image(X,p.outpath_grids,'grid',M.seed,p.prompt,opts.grid_format,info=M.info,short_filename=not opts.grid_extended_filename,grid=G,p=p)
		return M