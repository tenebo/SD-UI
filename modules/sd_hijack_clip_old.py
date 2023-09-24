from modules import sd_hijack_clip,shared
def process_text_old(self,texts):
	J=None;G=1.;C=self;Y=C.id_start;N=C.id_end;D=C.wrapped.max_length;O=[];P=[];Q=[];R=[];S=0;K={};Z=C.tokenize(texts);T=[]
	for H in Z:
		L=tuple(H)
		if L in K:A,I,B=K[L]
		else:
			I=[];A=[];B=[];M=G;E=0
			while E<len(H):
				U=H[E];F,a=C.hijack.embedding_db.find_embedding_at_position(H,E);V=C.token_mults.get(U)if shared.opts.enable_emphasis else J
				if V is not J:M*=V;E+=1
				elif F is J:A.append(U);B.append(M);E+=1
				else:W=int(F.vec.shape[0]);I.append((len(A),F));A+=[0]*W;B+=[M]*W;O.append((F.name,F.checksum()));E+=a
			if len(A)>D-2:b={B:A for(A,B)in C.wrapped.tokenizer.get_vocab().items()};c=A[D-2:];X=[b.get(int(A),'')for A in c];d=C.wrapped.tokenizer.convert_tokens_to_string(''.join(X));Q.append(f"Warning: too many input tokens; some ({len(X)}) have been truncated:\n{d}\n")
			S=len(A);A=A+[N]*(D-2-len(A));A=[Y]+A[0:D-2]+[N];K[L]=A,I,B
		B=B+[G]*(D-2-len(B));B=[G]+B[0:D-2]+[G];P.append(A);R.append(I);T.append(B)
	return T,P,O,Q,R,S
def forward_old(self,texts):
	A=self;C,D,B,E,F,H=process_text_old(A,texts);A.hijack.comments+=E
	if B:G=', '.join(f"{A} [{B}]"for(A,B)in B);A.hijack.comments.append(f"Used embeddings: {G}")
	A.hijack.fixes=F;return A.process_tokens(D,C)