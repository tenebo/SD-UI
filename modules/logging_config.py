import os,logging
def setup_logging(loglevel):
	A=loglevel
	if A is None:A=os.environ.get('SD_WEBUI_LOG_LEVEL')
	if A:B=getattr(logging,A.upper(),None)or logging.INFO;logging.basicConfig(level=B,format='%(asctime)s %(levelname)s [%(name)s] %(message)s',datefmt='%Y-%m-%d %H:%M:%S')