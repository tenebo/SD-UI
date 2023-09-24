_A=False
import time,argparse
class TimerSubcategory:
	def __init__(A,timer,category):B=timer;A.timer=B;A.category=category;A.start=None;A.original_base_category=B.base_category
	def __enter__(A):
		A.start=time.time();A.timer.base_category=A.original_base_category+A.category+'/';A.timer.subcategory_level+=1
		if A.timer.print_log:print(f"{'  '*A.timer.subcategory_level}{A.category}:")
	def __exit__(A,exc_type,exc_val,exc_tb):B=time.time()-A.start;A.timer.base_category=A.original_base_category;A.timer.add_time_to_record(A.original_base_category+A.category,B);A.timer.subcategory_level-=1;A.timer.record(A.category,disable_log=True)
class Timer:
	def __init__(A,print_log=_A):A.start=time.time();A.records={};A.total=0;A.base_category='';A.print_log=print_log;A.subcategory_level=0
	def elapsed(A):B=time.time();C=B-A.start;A.start=B;return C
	def add_time_to_record(A,category,amount):
		B=category
		if B not in A.records:A.records[B]=0
		A.records[B]+=amount
	def record(A,category,extra_time=0,disable_log=_A):
		D=category;B=extra_time;C=A.elapsed();A.add_time_to_record(A.base_category+D,C+B);A.total+=C+B
		if A.print_log and not disable_log:print(f"{'  '*A.subcategory_level}{D}: done in {C+B:.3f}s")
	def subcategory(A,name):A.elapsed();B=TimerSubcategory(A,name);return B
	def summary(B):
		A=f"{B.total:.1f}s";C=[(A,B)for(A,B)in B.records.items()if B>=.1 and'/'not in A]
		if not C:return A
		A+=' (';A+=', '.join([f"{A}: {B:.1f}s"for(A,B)in C]);A+=')';return A
	def dump(A):return{'total':A.total,'records':A.records}
	def reset(A):A.__init__()
parser=argparse.ArgumentParser(add_help=_A)
parser.add_argument('--log-startup',action='store_true',help="print a detailed log of what's happening at startup")
args=parser.parse_known_args()[0]
startup_timer=Timer(print_log=args.log_startup)
startup_record=None