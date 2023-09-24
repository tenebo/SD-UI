from PIL import Image,ImageFilter,ImageOps
def get_crop_region(mask,pad=0):
	'finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.\n    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)';C=pad;B=mask;D,E=B.shape;F=0
	for A in range(E):
		if not(B[:,A]==0).all():break
		F+=1
	G=0
	for A in reversed(range(E)):
		if not(B[:,A]==0).all():break
		G+=1
	H=0
	for A in range(D):
		if not(B[A]==0).all():break
		H+=1
	I=0
	for A in reversed(range(D)):
		if not(B[A]==0).all():break
		I+=1
	return int(max(F-C,0)),int(max(H-C,0)),int(min(E-G+C,E)),int(min(D-I+C,D))
def expand_crop_region(crop_region,processing_width,processing_height,image_width,image_height):
	'expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region\n    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128.';G=image_height;F=image_width;C,D,A,B=crop_region;K=(A-C)/(B-D);H=processing_width/processing_height
	if K>H:
		L=(A-C)/H;I=int(L-(B-D));D-=I//2;B+=I-I//2
		if B>=G:E=B-G;B-=E;D-=E
		if D<0:B-=D;D-=D
		if B>=G:B=G
	else:
		M=(B-D)*H;J=int(M-(A-C));C-=J//2;A+=J-J//2
		if A>=F:E=A-F;A-=E;C-=E
		if C<0:A-=C;C-=C
		if A>=F:A=F
	return C,D,A,B
def fill(image,mask):
	'fills masked regions with colors from image using blur. Not extremely effective.';D='RGBa';C='RGBA';A=image;E=Image.new(C,(A.width,A.height));B=Image.new(D,(A.width,A.height));B.paste(A.convert(C).convert(D),mask=ImageOps.invert(mask.convert('L')));B=B.convert(D)
	for(F,G)in[(256,1),(64,1),(16,2),(4,4),(2,2),(0,1)]:
		H=B.filter(ImageFilter.GaussianBlur(F)).convert(C)
		for I in range(G):E.alpha_composite(H)
	return E.convert('RGB')