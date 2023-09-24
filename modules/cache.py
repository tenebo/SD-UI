_B='cache.json'
_A=None
import json,os,os.path,threading,time
from modules.paths import data_path,script_path
cache_filename=os.environ.get('SD_WEBUI_CACHE_FILE',os.path.join(data_path,_B))
cache_data=_A
cache_lock=threading.Lock()
dump_cache_after=_A
dump_cache_thread=_A
def dump_cache():
	'\n    Marks cache for writing to disk. 5 seconds after no one else flags the cache for writing, it is written.\n    ';global dump_cache_after;global dump_cache_thread
	def A():
		global dump_cache_after;global dump_cache_thread
		while dump_cache_after is not _A and time.time()<dump_cache_after:time.sleep(1)
		with cache_lock:
			A=cache_filename+'-'
			with open(A,'w',encoding='utf8')as B:json.dump(cache_data,B,indent=4)
			os.replace(A,cache_filename);dump_cache_after=_A;dump_cache_thread=_A
	with cache_lock:
		dump_cache_after=time.time()+5
		if dump_cache_thread is _A:dump_cache_thread=threading.Thread(name='cache-writer',target=A);dump_cache_thread.start()
def cache(subsection):
	'\n    Retrieves or initializes a cache for a specific subsection.\n\n    Parameters:\n        subsection (str): The subsection identifier for the cache.\n\n    Returns:\n        dict: The cache data for the specified subsection.\n    ';A=subsection;global cache_data
	if cache_data is _A:
		with cache_lock:
			if cache_data is _A:
				if not os.path.isfile(cache_filename):cache_data={}
				else:
					try:
						with open(cache_filename,'r',encoding='utf8')as C:cache_data=json.load(C)
					except Exception:os.replace(cache_filename,os.path.join(script_path,'tmp',_B));print('[ERROR] issue occurred while trying to read cache.json, move current cache to tmp/cache.json and create new cache');cache_data={}
	B=cache_data.get(A,{});cache_data[A]=B;return B
def cached_data_for_file(subsection,title,filename,func):
	'\n    Retrieves or generates data for a specific file, using a caching mechanism.\n\n    Parameters:\n        subsection (str): The subsection of the cache to use.\n        title (str): The title of the data entry in the subsection of the cache.\n        filename (str): The path to the file to be checked for modifications.\n        func (callable): A function that generates the data if it is not available in the cache.\n\n    Returns:\n        dict or None: The cached or generated data, or None if data generation fails.\n\n    The `cached_data_for_file` function implements a caching mechanism for data stored in files.\n    It checks if the data associated with the given `title` is present in the cache and compares the\n    modification time of the file with the cached modification time. If the file has been modified,\n    the cache is considered invalid and the data is regenerated using the provided `func`.\n    Otherwise, the cached data is returned.\n\n    If the data generation fails, None is returned to indicate the failure. Otherwise, the generated\n    or cached data is returned as a dictionary.\n    ';C='mtime';D=title;B='value';E=cache(subsection);F=os.path.getmtime(filename);A=E.get(D)
	if A:
		H=A.get(C,0)
		if F>H:A=_A
	if not A or B not in A:
		G=func()
		if G is _A:return
		A={C:F,B:G};E[D]=A;dump_cache()
	return A[B]