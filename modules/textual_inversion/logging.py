import datetime,json,os
saved_params_shared={'batch_size','clip_grad_mode','clip_grad_value','create_image_every','data_root','gradient_step','initial_step','latent_sampling_method','learn_rate','log_directory','model_hash','model_name','num_of_dataset_images','steps','template_file','training_height','training_width'}
saved_params_ti={'embedding_name','num_vectors_per_token','save_embedding_every','save_image_with_stored_embedding'}
saved_params_hypernet={'activation_func','add_layer_norm','hypernetwork_name','layer_structure','save_hypernetwork_every','use_dropout','weight_init'}
saved_params_all=saved_params_shared|saved_params_ti|saved_params_hypernet
saved_params_previews={'preview_cfg_scale','preview_height','preview_negative_prompt','preview_prompt','preview_sampler_index','preview_seed','preview_steps','preview_width'}
def save_settings_to_file(log_directory,all_params):
	B=all_params;C=datetime.datetime.now();D={'datetime':C.strftime('%Y-%m-%d %H:%M:%S')};A=saved_params_all
	if B.get('preview_from_txt2img'):A=A|saved_params_previews
	D.update({B:C for(B,C)in B.items()if B in A});E=f"settings-{C.strftime('%Y-%m-%d-%H-%M-%S')}.json"
	with open(os.path.join(log_directory,E),'w')as F:json.dump(D,F,indent=4)