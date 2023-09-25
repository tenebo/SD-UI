from collections import defaultdict
def patch(key,obj,field,replacement):
	'Replaces a function in a module or a class.\n\n    Also stores the original function in this module, possible to be retrieved via original(key, obj, field).\n    If the function is already replaced by this caller (key), an exception is raised -- use undo() before that.\n\n    Arguments:\n        key: identifying information for who is doing the replacement. You can use __name__.\n        obj: the module or the class\n        field: name of the function as a string\n        replacement: the new function\n\n    Returns:\n        the original function\n    ';B=obj;A=field;C=B,A
	if C in originals[key]:raise RuntimeError(f"patch for {A} is already applied")
	D=getattr(B,A);originals[key][C]=D;setattr(B,A,replacement);return D
def undo(key,obj,field):
	'Undoes the peplacement by the patch().\n\n    If the function is not replaced, raises an exception.\n\n    Arguments:\n        key: identifying information for who is doing the replacement. You can use __name__.\n        obj: the module or the class\n        field: name of the function as a string\n\n    Returns:\n        Always None\n    ';A=field;B=obj,A
	if B not in originals[key]:raise RuntimeError(f"there is no patch for {A} to undo")
	C=originals[key].pop(B);setattr(obj,A,C)
def original(key,obj,field):'Returns the original function for the patch created by the patch() function';A=obj,field;return originals[key].get(A,None)
originals=defaultdict(dict)