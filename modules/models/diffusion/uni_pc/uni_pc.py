_K='time_uniform'
_J='logSNR'
_I='uncond'
_H='noise'
_G='cosine'
_F='linear'
_E=False
_D='discrete'
_C=True
_B=1.
_A=None
import torch,math,tqdm
class NoiseScheduleVP:
	def __init__(A,schedule=_D,betas=_A,alphas_cumprod=_A,continuous_beta_0=.1,continuous_beta_1=2e1):
		"Create a wrapper class for the forward SDE (VP type).\n\n        ***\n        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.\n                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.\n        ***\n\n        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).\n        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).\n        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:\n\n            log_alpha_t = self.marginal_log_mean_coeff(t)\n            sigma_t = self.marginal_std(t)\n            lambda_t = self.marginal_lambda(t)\n\n        Moreover, as lambda(t) is an invertible function, we also support its inverse function:\n\n            t = self.inverse_lambda(lambda_t)\n\n        ===============================================================\n\n        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).\n\n        1. For discrete-time DPMs:\n\n            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:\n                t_i = (i + 1) / N\n            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.\n            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.\n\n            Args:\n                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)\n                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)\n\n            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.\n\n            **Important**:  Please pay special attention for the args for `alphas_cumprod`:\n                The `alphas_cumprod` is the \\hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that\n                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \\sqrt{\\hat{alpha_n}} * x_0, (1 - \\hat{alpha_n}) * I ).\n                Therefore, the notation \\hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have\n                    alpha_{t_n} = \\sqrt{\\hat{alpha_n}},\n                and\n                    log(alpha_{t_n}) = 0.5 * log(\\hat{alpha_n}).\n\n\n        2. For continuous-time DPMs:\n\n            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise\n            schedule are the default settings in DDPM and improved-DDPM:\n\n            Args:\n                beta_min: A `float` number. The smallest beta for the linear schedule.\n                beta_max: A `float` number. The largest beta for the linear schedule.\n                cosine_s: A `float` number. The hyperparameter in the cosine schedule.\n                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.\n                T: A `float` number. The ending time of the forward process.\n\n        ===============================================================\n\n        Args:\n            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,\n                    'linear' or 'cosine' for continuous-time DPMs.\n        Returns:\n            A wrapper object of the forward SDE (VP type).\n\n        ===============================================================\n\n        Example:\n\n        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):\n        >>> ns = NoiseScheduleVP('discrete', betas=betas)\n\n        # For discrete-time DPMs, given alphas_cumprod (the \\hat{alpha_n} array for n = 0, 1, ..., N - 1):\n        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)\n\n        # For continuous-time DPMs (VPSDE), linear schedule:\n        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)\n\n        ";E=alphas_cumprod;D=betas;B=schedule
		if B not in[_D,_F,_G]:raise ValueError(f"Unsupported noise schedule {B}. The schedule needs to be 'discrete' or 'linear' or 'cosine'")
		A.schedule=B
		if B==_D:
			if D is not _A:C=.5*torch.log(1-D).cumsum(dim=0)
			else:assert E is not _A;C=.5*torch.log(E)
			A.total_N=len(C);A.T=_B;A.t_array=torch.linspace(.0,_B,A.total_N+1)[1:].reshape((1,-1));A.log_alpha_array=C.reshape((1,-1))
		else:
			A.total_N=1000;A.beta_0=continuous_beta_0;A.beta_1=continuous_beta_1;A.cosine_s=.008;A.cosine_beta_max=999.;A.cosine_t_max=math.atan(A.cosine_beta_max*(_B+A.cosine_s)/math.pi)*2.*(_B+A.cosine_s)/math.pi-A.cosine_s;A.cosine_log_alpha_0=math.log(math.cos(A.cosine_s/(_B+A.cosine_s)*math.pi/2.));A.schedule=B
			if B==_G:A.T=.9946
			else:A.T=_B
	def marginal_log_mean_coeff(A,t):
		'\n        Compute log(alpha_t) of a given continuous-time label t in [0, T].\n        '
		if A.schedule==_D:return interpolate_fn(t.reshape((-1,1)),A.t_array.to(t.device),A.log_alpha_array.to(t.device)).reshape(-1)
		elif A.schedule==_F:return-.25*t**2*(A.beta_1-A.beta_0)-.5*t*A.beta_0
		elif A.schedule==_G:B=lambda s:torch.log(torch.cos((s+A.cosine_s)/(_B+A.cosine_s)*math.pi/2.));C=B(t)-A.cosine_log_alpha_0;return C
	def marginal_alpha(A,t):'\n        Compute alpha_t of a given continuous-time label t in [0, T].\n        ';return torch.exp(A.marginal_log_mean_coeff(t))
	def marginal_std(A,t):'\n        Compute sigma_t of a given continuous-time label t in [0, T].\n        ';return torch.sqrt(_B-torch.exp(2.*A.marginal_log_mean_coeff(t)))
	def marginal_lambda(B,t):'\n        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].\n        ';A=B.marginal_log_mean_coeff(t);C=.5*torch.log(_B-torch.exp(2.*A));return A-C
	def inverse_lambda(A,lamb):
		'\n        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.\n        ';B=lamb
		if A.schedule==_F:E=2.*(A.beta_1-A.beta_0)*torch.logaddexp(-2.*B,torch.zeros((1,)).to(B));F=A.beta_0**2+E;return E/(torch.sqrt(F)+A.beta_0)/(A.beta_1-A.beta_0)
		elif A.schedule==_D:C=-.5*torch.logaddexp(torch.zeros((1,)).to(B.device),-2.*B);D=interpolate_fn(C.reshape((-1,1)),torch.flip(A.log_alpha_array.to(B.device),[1]),torch.flip(A.t_array.to(B.device),[1]));return D.reshape((-1,))
		else:C=-.5*torch.logaddexp(-2.*B,torch.zeros((1,)).to(B));G=lambda log_alpha_t:torch.arccos(torch.exp(log_alpha_t+A.cosine_log_alpha_0))*2.*(_B+A.cosine_s)/math.pi-A.cosine_s;D=G(C);return D
def model_wrapper(model,noise_schedule,model_type=_H,model_kwargs=_A,guidance_type=_I,guidance_scale=_B,classifier_fn=_A,classifier_kwargs=_A):
	'Create a wrapper function for the noise prediction model.\n\n    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to\n    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.\n\n    We support four types of the diffusion model by setting `model_type`:\n\n        1. "noise": noise prediction model. (Trained by predicting noise).\n\n        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).\n\n        3. "v": velocity prediction model. (Trained by predicting the velocity).\n            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].\n\n            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."\n                arXiv preprint arXiv:2202.00512 (2022).\n            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."\n                arXiv preprint arXiv:2210.02303 (2022).\n\n        4. "score": marginal score function. (Trained by denoising score matching).\n            Note that the score function and the noise prediction model follows a simple relationship:\n            ```\n                noise(x_t, t) = -sigma_t * score(x_t, t)\n            ```\n\n    We support three types of guided sampling by DPMs by setting `guidance_type`:\n        1. "uncond": unconditional sampling by DPMs.\n            The input `model` has the following format:\n            ``\n                model(x, t_input, **model_kwargs) -> noise | x_start | v | score\n            ``\n\n        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.\n            The input `model` has the following format:\n            ``\n                model(x, t_input, **model_kwargs) -> noise | x_start | v | score\n            ``\n\n            The input `classifier_fn` has the following format:\n            ``\n                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)\n            ``\n\n            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"\n                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.\n\n        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.\n            The input `model` has the following format:\n            ``\n                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score\n            ``\n            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.\n\n            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."\n                arXiv preprint arXiv:2207.12598 (2022).\n\n\n    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)\n    or continuous-time labels (i.e. epsilon to T).\n\n    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:\n    ``\n        def model_fn(x, t_continuous) -> noise:\n            t_input = get_model_input_time(t_continuous)\n            return noise_pred(model, x, t_input, **model_kwargs)\n    ``\n    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.\n\n    ===============================================================\n\n    Args:\n        model: A diffusion model with the corresponding format described above.\n        noise_schedule: A noise schedule object, such as NoiseScheduleVP.\n        model_type: A `str`. The parameterization type of the diffusion model.\n                    "noise" or "x_start" or "v" or "score".\n        model_kwargs: A `dict`. A dict for the other inputs of the model function.\n        guidance_type: A `str`. The type of the guidance for sampling.\n                    "uncond" or "classifier" or "classifier-free".\n        condition: A pytorch tensor. The condition for the guided sampling.\n                    Only used for "classifier" or "classifier-free" guidance type.\n        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.\n                    Only used for "classifier-free" guidance type.\n        guidance_scale: A `float`. The scale for the guided sampling.\n        classifier_fn: A classifier function. Only used for the classifier guidance.\n        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.\n    Returns:\n        A noise prediction model that accepts the noised data and the continuous time as the inputs.\n    ';P='classifier-free';O='classifier';N='x_start';L=classifier_fn;K=model;A=classifier_kwargs;J=guidance_scale;H=guidance_type;G=model_kwargs;D=model_type;C=noise_schedule;G=G or{};A=A or{}
	def M(t_continuous):
		'\n        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.\n        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].\n        For continuous-time DPMs, we just use `t_continuous`.\n        ';A=t_continuous
		if C.schedule==_D:return(A-_B/C.total_N)*1e3
		else:return A
	def I(x,t_continuous,cond=_A):
		A=t_continuous
		if A.reshape((-1,)).shape[0]==1:A=A.expand(x.shape[0])
		I=M(A)
		if cond is _A:E=K(x,I,_A,**G)
		else:E=K(x,I,cond,**G)
		if D==_H:return E
		elif D==N:H,F=C.marginal_alpha(A),C.marginal_std(A);B=x.dim();return(x-expand_dims(H,B)*E)/expand_dims(F,B)
		elif D=='v':H,F=C.marginal_alpha(A),C.marginal_std(A);B=x.dim();return expand_dims(H,B)*E+expand_dims(F,B)*x
		elif D=='score':F=C.marginal_std(A);B=x.dim();return-expand_dims(F,B)*E
	def R(x,t_input,condition):
		'\n        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).\n        '
		with torch.enable_grad():B=x.detach().requires_grad_(_C);C=L(B,t_input,condition,**A);return torch.autograd.grad(C.sum(),B)[0]
	def B(x,t_continuous,condition,unconditional_condition):
		'\n        The noise predicition model function that is used for DPM-Solver.\n        ';E=unconditional_condition;B=t_continuous;A=condition
		if B.reshape((-1,)).shape[0]==1:B=B.expand(x.shape[0])
		if H==_I:return I(x,B)
		elif H==O:assert L is not _A;S=M(B);K=R(x,S,A);T=C.marginal_std(B);G=I(x,B);return G-J*expand_dims(T,dims=K.dim())*K
		elif H==P:
			if J==_B or E is _A:return I(x,B,cond=A)
			else:
				U=torch.cat([x]*2);V=torch.cat([B]*2)
				if isinstance(A,dict):
					assert isinstance(E,dict);F={}
					for D in A:
						if isinstance(A[D],list):F[D]=[torch.cat([E[D][B],A[D][B]])for B in range(len(A[D]))]
						else:F[D]=torch.cat([E[D],A[D]])
				elif isinstance(A,list):
					F=[];assert isinstance(E,list)
					for N in range(len(A)):F.append(torch.cat([E[N],A[N]]))
				else:F=torch.cat([E,A])
				Q,G=I(U,V,cond=F).chunk(2);return Q+J*(G-Q)
	assert D in[_H,N,'v'];assert H in[_I,O,P];return B
class UniPC:
	def __init__(A,model_fn,noise_schedule,predict_x0=_C,thresholding=_E,max_val=_B,variant='bh1',condition=_A,unconditional_condition=_A,before_sample=_A,after_sample=_A,after_update=_A):'Construct a UniPC.\n\n        We support both data_prediction and noise_prediction.\n        ';A.model_fn_=model_fn;A.noise_schedule=noise_schedule;A.variant=variant;A.predict_x0=predict_x0;A.thresholding=thresholding;A.max_val=max_val;A.condition=condition;A.unconditional_condition=unconditional_condition;A.before_sample=before_sample;A.after_sample=after_sample;A.after_update=after_update
	def dynamic_thresholding_fn(C,x0,t=_A):'\n        The dynamic thresholding method.\n        ';B=x0;D=B.dim();E=C.dynamic_thresholding_ratio;A=torch.quantile(torch.abs(B).reshape((B.shape[0],-1)),E,dim=1);A=expand_dims(torch.maximum(A,C.thresholding_max_val*torch.ones_like(A).to(A.device)),D);B=torch.clamp(B,-A,A)/A;return B
	def model(A,x,t):
		C=A.condition;D=A.unconditional_condition
		if A.before_sample is not _A:x,t,C,D=A.before_sample(x,t,C,D)
		B=A.model_fn_(x,t,C,D)
		if A.after_sample is not _A:x,t,C,D,B=A.after_sample(x,t,C,D,B)
		if isinstance(B,tuple):B=B[1]
		return B
	def noise_prediction_fn(A,x,t):'\n        Return the noise prediction model.\n        ';return A.model(x,t)
	def data_prediction_fn(B,x,t):
		'\n        Return the data prediction model (with thresholding).\n        ';E=B.noise_prediction_fn(x,t);D=x.dim();F,G=B.noise_schedule.marginal_alpha(t),B.noise_schedule.marginal_std(t);C=(x-expand_dims(G,D)*E)/expand_dims(F,D)
		if B.thresholding:H=.995;A=torch.quantile(torch.abs(C).reshape((C.shape[0],-1)),H,dim=1);A=expand_dims(torch.maximum(A,B.max_val*torch.ones_like(A).to(A.device)),D);C=torch.clamp(C,-A,A)/A
		return C
	def model_fn(A,x,t):
		'\n        Convert the model to the noise prediction model or the data prediction model.\n        '
		if A.predict_x0:return A.data_prediction_fn(x,t)
		else:return A.noise_prediction_fn(x,t)
	def get_time_steps(C,skip_type,t_T,t_0,N,device):
		'Compute the intermediate time steps for sampling.\n        ';E=t_0;D=t_T;B=skip_type;A=device
		if B==_J:G=C.noise_schedule.marginal_lambda(torch.tensor(D).to(A));H=C.noise_schedule.marginal_lambda(torch.tensor(E).to(A));I=torch.linspace(G.cpu().item(),H.cpu().item(),N+1).to(A);return C.noise_schedule.inverse_lambda(I)
		elif B==_K:return torch.linspace(D,E,N+1).to(A)
		elif B=='time_quadratic':F=2;J=torch.linspace(D**(_B/F),E**(_B/F),N+1).pow(F).to(A);return J
		else:raise ValueError(f"Unsupported skip_type {B}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'")
	def get_orders_and_timesteps_for_singlestep_solver(G,steps,order,skip_type,t_T,t_0,device):
		'\n        Get the order of each step for sampling by the singlestep DPM-Solver.\n        ';F=device;E=skip_type;D=order;A=steps
		if D==3:
			B=A//3+1
			if A%3==0:C=[3]*(B-2)+[2,1]
			elif A%3==1:C=[3]*(B-1)+[1]
			else:C=[3]*(B-1)+[2]
		elif D==2:
			if A%2==0:B=A//2;C=[2]*B
			else:B=A//2+1;C=[2]*(B-1)+[1]
		elif D==1:B=A;C=[1]*A
		else:raise ValueError("'order' must be '1' or '2' or '3'.")
		if E==_J:H=G.get_time_steps(E,t_T,t_0,B,F)
		else:H=G.get_time_steps(E,t_T,t_0,A,F)[torch.cumsum(torch.tensor([0]+C),0).to(F)]
		return H,C
	def denoise_to_zero_fn(A,x,s):'\n        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.\n        ';return A.data_prediction_fn(x,s)
	def multistep_uni_pc_update(A,x,model_prev_list,t_prev_list,t,order,**E):
		D=order;C=t_prev_list;B=model_prev_list
		if len(t.shape)==0:t=t.view(-1)
		if'bh'in A.variant:return A.multistep_uni_pc_bh_update(x,B,C,t,D,**E)
		else:assert A.variant=='vary_coeff';return A.multistep_uni_pc_vary_update(x,B,C,t,D,**E)
	def multistep_uni_pc_vary_update(H,x,model_prev_list,t_prev_list,t,order,use_corrector=_C):
		a=order;Z=t_prev_list;R=use_corrector;Q=model_prev_list;P='bkchw,k->bchw';D=H.noise_schedule;assert a<=len(Q);S=Z[-1];b=D.marginal_lambda(S);h=D.marginal_lambda(t);I=Q[-1];i,J=D.marginal_std(S),D.marginal_std(t);T=D.marginal_log_mean_coeff(t);N=torch.exp(T);U=h-b;F=[];C=[]
		for c in range(1,a):j=Z[-(c+1)];k=Q[-(c+1)];l=D.marginal_lambda(j);d=(l-b)/U;F.append(d);C.append((k-I)/d)
		F.append(_B);F=torch.tensor(F,device=x.device);E=len(F);K=[];V=torch.ones_like(F)
		for A in range(1,E+1):K.append(V);V=V*F/(A+1)
		K=torch.stack(K,dim=1)
		if len(C)>0:C=torch.stack(C,dim=1);m=torch.linalg.inv(K[:-1,:-1]);e=m
		if R:n=torch.linalg.inv(K);O=n
		f=-U if H.predict_x0 else U;W=torch.expm1(f);G=[];g=1;X=W
		for A in range(1,E+2):G.append(X);X=X/f-1/g;g*=A+1
		L=_A
		if H.predict_x0:
			M=J/i*x-N*W*I;B=M
			if len(C)>0:
				for A in range(E-1):B=B-N*G[A+1]*torch.einsum(P,C,e[A])
			if R:
				L=H.model_fn(B,t);Y=L-I;B=M;A=0
				for A in range(E-1):B=B-N*G[A+1]*torch.einsum(P,C,O[A][:-1])
				B=B-N*G[E]*(Y*O[A][-1])
		else:
			o,T=D.marginal_log_mean_coeff(S),D.marginal_log_mean_coeff(t);M=torch.exp(T-o)*x-J*W*I;B=M
			if len(C)>0:
				for A in range(E-1):B=B-J*G[A+1]*torch.einsum(P,C,e[A])
			if R:
				L=H.model_fn(B,t);Y=L-I;B=M;A=0
				for A in range(E-1):B=B-J*G[A+1]*torch.einsum(P,C,O[A][:-1])
				B=B-J*G[E]*(Y*O[A][-1])
		return B,L
	def multistep_uni_pc_bh_update(D,x,model_prev_list,t_prev_list,t,order,x_t=_A,use_corrector=_C):
		e=t_prev_list;V=use_corrector;U=model_prev_list;T='k,bkchw->bchw';H=order;B=x_t;E=D.noise_schedule;assert H<=len(U);C=x.dim();W=e[-1];f=E.marginal_lambda(W);j=E.marginal_lambda(t);I=U[-1];k,S=E.marginal_std(W),E.marginal_std(t);l,g=E.marginal_log_mean_coeff(W),E.marginal_log_mean_coeff(t);X=torch.exp(g);Y=j-f;J=[];A=[]
		for K in range(1,H):m=e[-(K+1)];n=U[-(K+1)];o=E.marginal_lambda(m);h=((o-f)/Y)[0];J.append(h);A.append((n-I)/h)
		J.append(_B);J=torch.tensor(J,device=x.device);L=[];F=[];M=-Y[0]if D.predict_x0 else Y[0];Z=torch.expm1(M);a=Z/M-1;b=1
		if D.variant=='bh1':G=M
		elif D.variant=='bh2':G=torch.expm1(M)
		else:raise NotImplementedError()
		for K in range(1,H+1):L.append(torch.pow(J,K-1));F.append(a*b/G);b*=K+1;a=a/M-1/b
		L=torch.stack(L);F=torch.tensor(F,device=x.device);i=len(A)>0 and B is _A
		if len(A)>0:
			A=torch.stack(A,dim=1)
			if B is _A:
				if H==2:c=torch.tensor([.5],device=F.device)
				else:c=torch.linalg.solve(L[:-1,:-1],F[:-1])
		else:A=_A
		if V:
			if H==1:N=torch.tensor([.5],device=F.device)
			else:N=torch.linalg.solve(L,F)
		O=_A
		if D.predict_x0:
			P=expand_dims(S/k,C)*x-expand_dims(X*Z,C)*I
			if B is _A:
				if i:Q=torch.einsum(T,c,A)
				else:Q=0
				B=P-expand_dims(X*G,C)*Q
			if V:
				O=D.model_fn(B,t)
				if A is not _A:R=torch.einsum(T,N[:-1],A)
				else:R=0
				d=O-I;B=P-expand_dims(X*G,C)*(R+N[-1]*d)
		else:
			P=expand_dims(torch.exp(g-l),C)*x-expand_dims(S*Z,C)*I
			if B is _A:
				if i:Q=torch.einsum(T,c,A)
				else:Q=0
				B=P-expand_dims(S*G,C)*Q
			if V:
				O=D.model_fn(B,t)
				if A is not _A:R=torch.einsum(T,N[:-1],A)
				else:R=0
				d=O-I;B=P-expand_dims(S*G,C)*(R+N[-1]*d)
		return B,O
	def sample(A,x,steps=20,t_start=_A,t_end=_A,order=3,skip_type=_K,method='singlestep',lower_order_final=_C,denoise_to_zero=_E,solver_type='dpm_solver',atol=.0078,rtol=.05,corrector=_E):
		L=t_end;K=t_start;E=order;D=steps;M=_B/A.noise_schedule.total_N if L is _A else L;S=A.noise_schedule.T if K is _A else K;N=x.device
		if method=='multistep':
			assert D>=E,'UniPC order must be < sampling steps';H=A.get_time_steps(skip_type=skip_type,t_T=S,t_0=M,N=D,device=N);assert H.shape[0]-1==D
			with torch.no_grad():
				B=H[0].expand(x.shape[0]);F=[A.model_fn(x,B)];G=[B]
				with tqdm.tqdm(total=D)as O:
					for P in range(1,E):
						B=H[P].expand(x.shape[0]);x,C=A.multistep_uni_pc_update(x,F,G,B,P,use_corrector=_C)
						if C is _A:C=A.model_fn(x,B)
						if A.after_update is not _A:A.after_update(x,C)
						F.append(C);G.append(B);O.update()
					for I in range(E,D+1):
						B=H[I].expand(x.shape[0])
						if lower_order_final:Q=min(E,D+1-I)
						else:Q=E
						if I==D:R=_E
						else:R=_C
						x,C=A.multistep_uni_pc_update(x,F,G,B,Q,use_corrector=R)
						if A.after_update is not _A:A.after_update(x,C)
						for J in range(E-1):G[J]=G[J+1];F[J]=F[J+1]
						G[-1]=B
						if I<D:
							if C is _A:C=A.model_fn(x,B)
							F[-1]=C
						O.update()
		else:raise NotImplementedError()
		if denoise_to_zero:x=A.denoise_to_zero_fn(x,torch.ones((x.shape[0],)).to(N)*M)
		return x
def interpolate_fn(x,xp,yp):'\n    A piecewise linear function y = f(x), using xp and yp as keypoints.\n    We implement f(x) in a differentiable way (i.e. applicable for autograd).\n    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)\n\n    Args:\n        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).\n        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.\n        yp: PyTorch tensor with shape [C, K].\n    Returns:\n        The function values f(x), with shape [N, C].\n    ';E,B=x.shape[0],xp.shape[1];K=torch.cat([x.unsqueeze(2),xp.unsqueeze(0).repeat((E,1,1))],dim=2);F,L=torch.sort(K,dim=2);A=torch.argmin(L,dim=2);D=A-1;C=torch.where(torch.eq(A,0),torch.tensor(1,device=x.device),torch.where(torch.eq(A,B),torch.tensor(B-2,device=x.device),D));M=torch.where(torch.eq(C,D),C+2,C+1);G=torch.gather(F,dim=2,index=C.unsqueeze(2)).squeeze(2);N=torch.gather(F,dim=2,index=M.unsqueeze(2)).squeeze(2);H=torch.where(torch.eq(A,0),torch.tensor(0,device=x.device),torch.where(torch.eq(A,B),torch.tensor(B-2,device=x.device),D));I=yp.unsqueeze(0).expand(E,-1,-1);J=torch.gather(I,dim=2,index=H.unsqueeze(2)).squeeze(2);O=torch.gather(I,dim=2,index=(H+1).unsqueeze(2)).squeeze(2);P=J+(x-G)*(O-J)/(N-G);return P
def expand_dims(v,dims):'\n    Expand the tensor `v` to the dim `dims`.\n\n    Args:\n        `v`: a PyTorch tensor with shape [N].\n        `dim`: a `int`.\n    Returns:\n        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.\n    ';return v[(...,)+(_A,)*(dims-1)]