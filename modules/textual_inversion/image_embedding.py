_D='string_to_param'
_C='RGBA'
_B='TORCHTENSOR'
_A='RGB'
import base64,json,warnings,numpy as np,zlib
from PIL import Image,ImageDraw
import torch
class EmbeddingEncoder(json.JSONEncoder):
	def default(B,obj):
		A=obj
		if isinstance(A,torch.Tensor):return{_B:A.cpu().detach().numpy().tolist()}
		return json.JSONEncoder.default(B,A)
class EmbeddingDecoder(json.JSONDecoder):
	def __init__(A,*B,**C):json.JSONDecoder.__init__(A,*B,object_hook=A.object_hook,**C)
	def object_hook(A,d):
		if _B in d:return torch.from_numpy(np.array(d[_B]))
		return d
def embedding_to_b64(data):A=json.dumps(data,cls=EmbeddingEncoder);return base64.b64encode(A.encode())
def embedding_from_b64(data):A=base64.b64decode(data);return json.loads(A,cls=EmbeddingDecoder)
def lcg(m=2**32,a=1664525,c=1013904223,seed=0):
	A=seed
	while True:A=(a*A+c)%m;yield A%255
def xor_block(block):A=block;B=lcg();C=np.array([next(B)for A in range(np.product(A.shape))]).astype(np.uint8).reshape(A.shape);return np.bitwise_xor(A.astype(np.uint8),C&15)
def style_block(block,sequence):
	E=sequence;B=block;A=Image.new(_A,(B.shape[1],B.shape[0]));I=ImageDraw.Draw(A);F=0
	for G in range(-6,A.size[0],8):
		for(J,H)in enumerate(range(-6,A.size[1],8)):
			C=0
			if J%2==0:C=4
			D=E[F%len(E)];F+=1;I.ellipse((G+C,H,G+6+C,H+6),fill=(D,D,D))
	K=np.array(A).astype(np.uint8)&240;return B^K
def insert_image_data_embed(image,data):D=image;G=3;L=zlib.compress(json.dumps(data,cls=EmbeddingEncoder).encode(),level=9);J=np.frombuffer(L,np.uint8).copy();B=J>>4;A=J&15;C=D.size[1];E=A.shape[0]+(C-A.shape[0]%C);E=E+(C*G-E%(C*G));A=np.resize(A,E);A=A.reshape((C,-1,G));B=np.resize(B,E);B=B.reshape((C,-1,G));F=list(data[_D].values())[0].cpu().detach().numpy().tolist()[0][:1024];F=(np.abs(F)/np.max(np.abs(F))*255).astype(np.uint8);A=style_block(A,sequence=F);A=xor_block(A);B=style_block(B,sequence=F[::-1]);B=xor_block(B);H=Image.fromarray(A,mode=_A);K=Image.fromarray(B,mode=_A);I=Image.new(_A,(D.size[0]+H.size[0]+K.size[0]+2,D.size[1]),(0,0,0));I.paste(H,(0,0));I.paste(D,(H.size[0]+1,0));I.paste(K,(H.size[0]+1+D.size[0]+1,0));return I
def crop_black(img,tol=0):A=(img>tol).all(2);B,C=A.any(0),A.any(1);D,E=B.argmax(),A.shape[1]-B[::-1].argmax();F,G=C.argmax(),A.shape[0]-C[::-1].argmax();return img[F:G,D:E]
def extract_image_data_embed(image):
	A=image;G=3;B=crop_black(np.array(A.convert(_A).getdata()).reshape(A.size[1],A.size[0],G).astype(np.uint8))&15;C=np.where(np.sum(B,axis=(0,2))==0)
	if C[0].shape[0]<2:print('No Image data blocks found.');return
	D=B[:,:C[0].min(),:].astype(np.uint8);E=B[:,C[0].max()+1:,:].astype(np.uint8);D=xor_block(D);E=xor_block(E);F=E<<4|D;F=F.flatten().tobytes();H=zlib.decompress(F);return json.loads(H,cls=EmbeddingDecoder)
def caption_image_overlay(srcimage,title,footerLeft,footerMid,footerRight,textfont=None):
	M=footerRight;N=footerMid;O=footerLeft;I=title;from modules.images import get_font as J
	if textfont:warnings.warn('passing in a textfont to caption_image_overlay is deprecated and does nothing',DeprecationWarning,stacklevel=2)
	from math import cos;A=srcimage.copy();G=32;P=1.5;Q=Image.new(_C,(1,A.size[1]),color=(0,0,0,0))
	for K in range(A.size[1]):L=1-cos(K/A.size[1]*P);L=max(L,1-cos((A.size[1]-K)/A.size[1]*P*1.1));Q.putpixel((0,K),(0,0,0,int(L*255)))
	A=Image.alpha_composite(A.convert(_C),Q.resize(A.size));D=ImageDraw.Draw(A);B=J(G);C=10;E,E,F,H=D.textbbox((0,0),I,font=B);G=min(int(G*((A.size[0]*.75-C*4)/F)),72);B=J(G);E,E,F,H=D.textbbox((0,0),I,font=B);D.text((C,C),I,anchor='lt',font=B,fill=(255,255,255,230));E,E,F,H=D.textbbox((0,0),O,font=B);R=min(int(G*((A.size[0]/3-C)/F)),72);E,E,F,H=D.textbbox((0,0),N,font=B);S=min(int(G*((A.size[0]/3-C)/F)),72);E,E,F,H=D.textbbox((0,0),M,font=B);T=min(int(G*((A.size[0]/3-C)/F)),72);B=J(min(R,S,T));D.text((C,A.size[1]-C),O,anchor='ls',font=B,fill=(255,255,255,230));D.text((A.size[0]/2,A.size[1]-C),N,anchor='ms',font=B,fill=(255,255,255,230));D.text((A.size[0]-C,A.size[1]-C),M,anchor='rs',font=B,fill=(255,255,255,230));return A
if __name__=='__main__':testEmbed=Image.open('test_embedding.png');data=extract_image_data_embed(testEmbed);assert data is not None;data=embedding_from_b64(testEmbed.text['sd-ti-embedding']);assert data is not None;image=Image.new(_C,(512,512),(255,255,200,255));cap_image=caption_image_overlay(image,'title','footerLeft','footerMid','footerRight');test_embed={_D:{'*':torch.from_numpy(np.random.random((2,4096)))}};embedded_image=insert_image_data_embed(cap_image,test_embed);retrived_embed=extract_image_data_embed(embedded_image);assert str(retrived_embed)==str(test_embed);embedded_image2=insert_image_data_embed(cap_image,retrived_embed);assert embedded_image==embedded_image2;g=lcg();shared_random=np.array([next(g)for A in range(100)]).astype(np.uint8).tolist();reference_random=[253,242,127,44,157,27,239,133,38,79,167,4,177,95,130,79,78,14,52,215,220,194,126,28,240,179,160,153,149,50,105,14,21,218,199,18,54,198,193,38,128,19,53,195,124,75,205,12,6,145,0,28,30,148,8,45,218,171,55,249,97,166,12,35,0,41,221,122,215,170,31,113,186,97,119,31,23,185,66,140,30,41,37,63,137,109,216,55,159,145,82,204,86,73,222,44,198,118,240,97];assert shared_random==reference_random;hunna_kay_random_sum=sum(np.array([next(g)for A in range(100000)]).astype(np.uint8).tolist());assert 12731374==hunna_kay_random_sum