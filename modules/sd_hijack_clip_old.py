from modules import sd_hijack_clip,shared
def process_text_old(self,texts):
	M=None;I=1.;C=self;Y=C.id_start;N=C.id_end;D=C.wrapped.max_length;O=[];P=[];Q=[];R=[];S=0;J={};Z=C.tokenize(texts);T=[]
	for G in Z:
		K=tuple(G)
		if K in J:A,H,B=J[K]
		else:
			H=[];A=[];B=[];L=I;E=0
			while E<len(G):
				U=G[E];F,a=C.hijack.embedding_db.find_embedding_at_position(G,E);V=C.token_mults.get(U)if shared.opts.enable_emphasis else M
				if V is not M:L*=V;E+=1
				elif F is M:A.append(U);B.append(L);E+=1
				else:W=int(F.vec.shape[0]);H.append((len(A),F));A+=[0]*W;B+=[L]*W;O.append((F.name,F.checksum()));E+=a
			if len(A)>D-2:b={B:A for(A,B)in C.wrapped.tokenizer.get_vocab().items()};c=A[D-2:];X=[b.get(int(A),'')for A in c];d=C.wrapped.tokenizer.convert_tokens_to_string(''.join(X));Q.append(f"Warning: too many input tokens; some ({len(X)}) have been truncated:\n{d}\n")
			S=len(A);A=A+[N]*(D-2-len(A));A=[Y]+A[0:D-2]+[N];J[K]=A,H,B
		B=B+[I]*(D-2-len(B));B=[I]+B[0:D-2]+[I];P.append(A);R.append(H);T.append(B)
	return T,P,O,Q,R,S
def forward_old(self,texts):
	A=self;C,D,B,E,F,H=process_text_old(A,texts);A.hijack.comments+=E
	if B:G=', '.join(f"{A} [{B}]"for(A,B)in B);A.hijack.comments.append(f"Used embeddings: {G}")
	A.hijack.fixes=F;return A.process_tokens(D,C)