_B='finished'
_A='interrupted'
import html,gradio as gr,modules.textual_inversion.textual_inversion,modules.textual_inversion.preprocess
from modules import sd_hijack,shared
def create_embedding(name,initialization_text,nvpt,overwrite_old):A=modules.textual_inversion.textual_inversion.create_embedding(name,nvpt,overwrite_old,init_text=initialization_text);sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings();return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())),f"Created: {A}",''
def preprocess(*A):modules.textual_inversion.preprocess.preprocess(*A);return f"Preprocessing {_A if shared.state.interrupted else _B}.",''
def train_embedding(*B):
	assert not shared.cmd_opts.lowvram,'Training models with lowvram not possible';A=shared.opts.training_xattention_optimizations
	try:
		if not A:sd_hijack.undo_optimizations()
		C,D=modules.textual_inversion.textual_inversion.train_embedding(*B);E=f"\nTraining {_A if shared.state.interrupted else _B} at {C.step} steps.\nEmbedding saved to {html.escape(D)}\n";return E,''
	except Exception:raise
	finally:
		if not A:sd_hijack.apply_optimizations()