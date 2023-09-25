import os,gradio as gr
from modules import errors,shared
from modules.paths_internal import script_path
gradio_hf_hub_themes=['gradio/base','gradio/glass','gradio/monochrome','gradio/seafoam','gradio/soft','gradio/dracula_test','abidlabs/dracula_test','abidlabs/Lime','abidlabs/pakistan','Ama434/neutral-barlow','dawood/microsoft_windows','finlaymacklon/smooth_slate','Franklisi/darkmode','freddyaboulton/dracula_revamped','freddyaboulton/test-blue','gstaff/xkcd','Insuz/Mocha','Insuz/SimpleIndigo','JohnSmith9982/small_and_pretty','nota-ai/theme','nuttea/Softblue','ParityError/Anime','reilnuud/polite','remilia/Ghostly','rottenlittlecreature/Moon_Goblin','step-3-profit/Midnight-Deep','Taithrah/Minimal','ysharma/huggingface','ysharma/steampunk','NoCrypt/miku']
def reload_gradio_theme(theme_name=None):
	A=theme_name
	if not A:A=shared.opts.gradio_theme
	C=dict(font=['Source Sans Pro','ui-sans-serif','system-ui','sans-serif'],font_mono=['IBM Plex Mono','ui-monospace','Consolas','monospace'])
	if A=='Default':shared.gradio_theme=gr.themes.Default(**C)
	else:
		try:
			D=os.path.join(script_path,'tmp','gradio_themes');B=os.path.join(D,f"{A.replace('/','_')}.json")
			if shared.opts.gradio_themes_cache and os.path.exists(B):shared.gradio_theme=gr.themes.ThemeClass.load(B)
			else:os.makedirs(D,exist_ok=True);shared.gradio_theme=gr.themes.ThemeClass.from_hub(A);shared.gradio_theme.dump(B)
		except Exception as E:errors.display(E,'changing gradio theme');shared.gradio_theme=gr.themes.Default(**C)