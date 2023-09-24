import modules.scripts as scripts,gradio as gr,ast,copy
from modules.processing import Processed
from modules.shared import cmd_opts
def convertExpr2Expression(expr):expr.lineno=0;expr.col_offset=0;result=ast.Expression(expr.value,lineno=0,col_offset=0);return result
def exec_with_return(code,module):
	'\n    like exec() but can return values\n    https://stackoverflow.com/a/52361938/5862977\n    ';B='exec';A='<ast>';code_ast=ast.parse(code);init_ast=copy.deepcopy(code_ast);init_ast.body=code_ast.body[:-1];last_ast=copy.deepcopy(code_ast);last_ast.body=code_ast.body[-1:];exec(compile(init_ast,A,B),module.__dict__)
	if type(last_ast.body[0])==ast.Expr:return eval(compile(convertExpr2Expression(last_ast.body[0]),A,'eval'),module.__dict__)
	else:exec(compile(last_ast,A,B),module.__dict__)
class Script(scripts.Script):
	def title(self):return'Custom code'
	def show(self,is_img2img):return cmd_opts.allow_code
	def ui(self,is_img2img):example='from modules.processing import process_images\n\np.width = 768\np.height = 768\np.batch_size = 2\np.steps = 10\n\nreturn process_images(p)\n';code=gr.Code(value=example,language='python',label='Python code',elem_id=self.elem_id('code'));indent_level=gr.Number(label='Indent level',value=2,precision=0,elem_id=self.elem_id('indent_level'));return[code,indent_level]
	def run(self,p,code,indent_level):
		assert cmd_opts.allow_code,'--allow-code option must be enabled';display_result_data=[[],-1,'']
		def display(imgs,s=display_result_data[1],i=display_result_data[2]):display_result_data[0]=imgs;display_result_data[1]=s;display_result_data[2]=i
		from types import ModuleType;module=ModuleType('testmodule');module.__dict__.update(globals());module.p=p;module.display=display;indent=' '*indent_level;indented=code.replace('\n',f"\n{indent}");body=f"def __ouruitemp__():\n{indent}{indented}\n__ouruitemp__()";result=exec_with_return(body,module)
		if isinstance(result,Processed):return result
		return Processed(p,*display_result_data)