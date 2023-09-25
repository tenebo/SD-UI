import ngrok
def connect(token,port,options):
	D='session_metadata';B=options;A=token;C=None
	if A is None:A='None'
	elif':'in A:A,E,F=A.split(':',2);C=f"{E}:{F}"
	if not B.get('authtoken_from_env'):B['authtoken']=A
	if C:B['basic_auth']=C
	if not B.get(D):B[D]='stable-diffusion-webui'
	try:G=ngrok.connect(f"127.0.0.1:{port}",**B).url()
	except Exception as H:print(f"Invalid ngrok authtoken? ngrok connection aborted due to: {H}\nYour token: {A}, get the right one on https://dashboard.ngrok.com/get-started/your-authtoken")
	else:print(f"ngrok connected to localhost:{port}! URL: {G}\nYou can use this link after the launch is complete.")