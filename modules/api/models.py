_W='Filename'
_V='Options'
_U='Images'
_T="Image to work on, must be a Base64 string containing the image's data."
_S='alwayson_scripts'
_R='save_images'
_Q='send_images'
_P='script_args'
_O='script_name'
_N='Minimum'
_M='Model Name'
_L='The generated image in base64 format.'
_K='exclude'
_J='sampler_index'
_I='Path'
_H=True
_G='Image'
_F='Name'
_E=False
_D='default'
_C='type'
_B='key'
_A=None
import inspect
from pydantic import BaseModel,Field,create_model
from typing import Any,Optional
from typing_extensions import Literal
from inflection import underscore
from modules.processing import StableDiffusionProcessingTxt2Img,StableDiffusionProcessingImg2Img
from modules.shared import sd_upscalers,opts,parser
from typing import Dict,List
API_NOT_ALLOWED=['self','kwargs','sd_model','outpath_samples','outpath_grids',_J,'extra_generation_params','overlay_images','do_not_reload_embeddings','seed_enable_extras','prompt_for_display','sampler_noise_scheduler_override','ddim_discretize']
class ModelDef(BaseModel):'Assistance Class for Pydantic Dynamic Model Generation';field:0;field_alias:0;field_type:0;field_value:0;field_exclude=_E
class PydanticModelGenerator:
	'\n    Takes in created classes and stubs them out in a way FastAPI/Pydantic is happy about:\n    source_data is a snapshot of the default values produced by the class\n    params are the names of the actual keys required by __init__\n    '
	def __init__(self,model_name=_A,class_instance=_A,additional_fields=_A):
		def field_type_generator(k,v):
			field_type=v.annotation
			if field_type==_G:field_type='str'
			return Optional[field_type]
		def merge_class_params(class_):
			all_classes=list(filter(lambda x:x is not object,inspect.getmro(class_)));parameters={}
			for classes in all_classes:parameters={**parameters,**inspect.signature(classes.__init__).parameters}
			return parameters
		self._model_name=model_name;self._class_data=merge_class_params(class_instance);self._model_def=[ModelDef(field=underscore(k),field_alias=k,field_type=field_type_generator(k,v),field_value=_A if isinstance(v.default,property)else v.default)for(k,v)in self._class_data.items()if k not in API_NOT_ALLOWED]
		for fields in additional_fields:self._model_def.append(ModelDef(field=underscore(fields[_B]),field_alias=fields[_B],field_type=fields[_C],field_value=fields[_D],field_exclude=fields[_K]if _K in fields else _E))
	def generate_model(self):'\n        Creates a pydantic BaseModel\n        from the json and overrides provided at initialization\n        ';fields={d.field:(d.field_type,Field(default=d.field_value,alias=d.field_alias,exclude=d.field_exclude))for d in self._model_def};DynamicModel=create_model(self._model_name,**fields);DynamicModel.__config__.allow_population_by_field_name=_H;DynamicModel.__config__.allow_mutation=_H;return DynamicModel
StableDiffusionTxt2ImgProcessingAPI=PydanticModelGenerator('StableDiffusionProcessingTxt2Img',StableDiffusionProcessingTxt2Img,[{_B:_J,_C:str,_D:'Euler'},{_B:_O,_C:str,_D:_A},{_B:_P,_C:list,_D:[]},{_B:_Q,_C:bool,_D:_H},{_B:_R,_C:bool,_D:_E},{_B:_S,_C:dict,_D:{}}]).generate_model()
StableDiffusionImg2ImgProcessingAPI=PydanticModelGenerator('StableDiffusionProcessingImg2Img',StableDiffusionProcessingImg2Img,[{_B:_J,_C:str,_D:'Euler'},{_B:'init_images',_C:list,_D:_A},{_B:'denoising_strength',_C:float,_D:.75},{_B:'mask',_C:str,_D:_A},{_B:'include_init_images',_C:bool,_D:_E,_K:_H},{_B:_O,_C:str,_D:_A},{_B:_P,_C:list,_D:[]},{_B:_Q,_C:bool,_D:_H},{_B:_R,_C:bool,_D:_E},{_B:_S,_C:dict,_D:{}}]).generate_model()
class TextToImageResponse(BaseModel):images=Field(default=_A,title=_G,description=_L);parameters:0;info:0
class ImageToImageResponse(BaseModel):images=Field(default=_A,title=_G,description=_L);parameters:0;info:0
class ExtrasBaseRequest(BaseModel):resize_mode=Field(default=0,title='Resize Mode',description='Sets the resize mode: 0 to upscale by upscaling_resize amount, 1 to upscale up to upscaling_resize_h x upscaling_resize_w.');show_extras_results=Field(default=_H,title='Show results',description='Should the backend return the generated image?');gfpgan_visibility=Field(default=0,title='GFPGAN Visibility',ge=0,le=1,allow_inf_nan=_E,description='Sets the visibility of GFPGAN, values should be between 0 and 1.');codeformer_visibility=Field(default=0,title='CodeFormer Visibility',ge=0,le=1,allow_inf_nan=_E,description='Sets the visibility of CodeFormer, values should be between 0 and 1.');codeformer_weight=Field(default=0,title='CodeFormer Weight',ge=0,le=1,allow_inf_nan=_E,description='Sets the weight of CodeFormer, values should be between 0 and 1.');upscaling_resize=Field(default=2,title='Upscaling Factor',ge=1,le=8,description='By how much to upscale the image, only used when resize_mode=0.');upscaling_resize_w=Field(default=512,title='Target Width',ge=1,description='Target width for the upscaler to hit. Only used when resize_mode=1.');upscaling_resize_h=Field(default=512,title='Target Height',ge=1,description='Target height for the upscaler to hit. Only used when resize_mode=1.');upscaling_crop=Field(default=_H,title='Crop to fit',description='Should the upscaler crop the image to fit in the chosen size?');upscaler_1=Field(default='None',title='Main upscaler',description=f"The name of the main upscaler to use, it has to be one of this list: {' , '.join([x.name for x in sd_upscalers])}");upscaler_2=Field(default='None',title='Secondary upscaler',description=f"The name of the secondary upscaler to use, it has to be one of this list: {' , '.join([x.name for x in sd_upscalers])}");extras_upscaler_2_visibility=Field(default=0,title='Secondary upscaler visibility',ge=0,le=1,allow_inf_nan=_E,description='Sets the visibility of secondary upscaler, values should be between 0 and 1.');upscale_first=Field(default=_E,title='Upscale first',description='Should the upscaler run before restoring faces?')
class ExtraBaseResponse(BaseModel):html_info=Field(title='HTML info',description='A series of HTML tags containing the process info.')
class ExtrasSingleImageRequest(ExtrasBaseRequest):image=Field(default='',title=_G,description=_T)
class ExtrasSingleImageResponse(ExtraBaseResponse):image=Field(default=_A,title=_G,description=_L)
class FileData(BaseModel):data=Field(title='File data',description='Base64 representation of the file');name=Field(title='File name')
class ExtrasBatchImagesRequest(ExtrasBaseRequest):imageList=Field(title=_U,description='List of images to work on. Must be Base64 strings')
class ExtrasBatchImagesResponse(ExtraBaseResponse):images=Field(title=_U,description='The generated images in base64 format.')
class PNGInfoRequest(BaseModel):image=Field(title=_G,description='The base64 encoded PNG image')
class PNGInfoResponse(BaseModel):info=Field(title='Image info',description='A string with the parameters used to generate the image');items=Field(title='Items',description='An object containing all the info the image had')
class ProgressRequest(BaseModel):skip_current_image=Field(default=_E,title='Skip current image',description='Skip current image serialization')
class ProgressResponse(BaseModel):progress=Field(title='Progress',description='The progress with a range of 0 to 1');eta_relative=Field(title='ETA in secs');state=Field(title='State',description='The current state snapshot');current_image=Field(default=_A,title='Current image',description='The current image in base64 format. opts.show_progress_every_n_steps is required for this to work.');textinfo=Field(default=_A,title='Info text',description='Info text used by OurUI.')
class InterrogateRequest(BaseModel):image=Field(default='',title=_G,description=_T);model=Field(default='clip',title='Model',description='The interrogate model used.')
class InterrogateResponse(BaseModel):caption=Field(default=_A,title='Caption',description='The generated caption for the image.')
class TrainResponse(BaseModel):info=Field(title='Train info',description='Response string from train embedding or hypernetwork task.')
class CreateResponse(BaseModel):info=Field(title='Create info',description='Response string from create embedding or hypernetwork task.')
class PreprocessResponse(BaseModel):info=Field(title='Preprocess info',description='Response string from preprocessing task.')
fields={}
for(key,metadata)in opts.data_labels.items():
	value=opts.data.get(key);optType=opts.typemap.get(type(metadata.default),type(metadata.default))if metadata.default else Any
	if metadata is not _A:fields.update({key:(Optional[optType],Field(default=metadata.default,description=metadata.label))})
	else:fields.update({key:(Optional[optType],Field())})
OptionsModel=create_model(_V,**fields)
flags={}
_options=vars(parser)['_option_string_actions']
for key in _options:
	if _options[key].dest!='help':
		flag=_options[key];_type=str
		if _options[key].default is not _A:_type=type(_options[key].default)
		flags.update({flag.dest:(_type,Field(default=flag.default,description=flag.help))})
FlagsModel=create_model('Flags',**flags)
class SamplerItem(BaseModel):name=Field(title=_F);aliases=Field(title='Aliases');options=Field(title=_V)
class UpscalerItem(BaseModel):name=Field(title=_F);model_name=Field(title=_M);model_path=Field(title=_I);model_url=Field(title='URL');scale=Field(title='Scale')
class LatentUpscalerModeItem(BaseModel):name=Field(title=_F)
class SDModelItem(BaseModel):title=Field(title='Title');model_name=Field(title=_M);hash=Field(title='Short hash');sha256=Field(title='sha256 hash');filename=Field(title=_W);config=Field(title='Config file')
class SDVaeItem(BaseModel):model_name=Field(title=_M);filename=Field(title=_W)
class HypernetworkItem(BaseModel):name=Field(title=_F);path=Field(title=_I)
class FaceRestorerItem(BaseModel):name=Field(title=_F);cmd_dir=Field(title=_I)
class RealesrganItem(BaseModel):name=Field(title=_F);path=Field(title=_I);scale=Field(title='Scale')
class PromptStyleItem(BaseModel):name=Field(title=_F);prompt=Field(title='Prompt');negative_prompt=Field(title='Negative Prompt')
class EmbeddingItem(BaseModel):step=Field(title='Step',description='The number of steps that were used to train this embedding, if available');sd_checkpoint=Field(title='SD Checkpoint',description='The hash of the checkpoint this embedding was trained on, if available');sd_checkpoint_name=Field(title='SD Checkpoint Name',description='The name of the checkpoint this embedding was trained on, if available. Note that this is the name that was used by the trainer; for a stable identifier, use `sd_checkpoint` instead');shape=Field(title='Shape',description='The length of each individual vector in the embedding');vectors=Field(title='Vectors',description='The number of vectors in the embedding')
class EmbeddingsResponse(BaseModel):loaded=Field(title='Loaded',description='Embeddings loaded for the current model');skipped=Field(title='Skipped',description='Embeddings skipped for the current model (likely due to architecture incompatibility)')
class MemoryResponse(BaseModel):ram=Field(title='RAM',description='System memory stats');cuda=Field(title='CUDA',description='nVidia CUDA memory stats')
class ScriptsList(BaseModel):txt2img=Field(default=_A,title='Txt2img',description='Titles of scripts (txt2img)');img2img=Field(default=_A,title='Img2img',description='Titles of scripts (img2img)')
class ScriptArg(BaseModel):label=Field(default=_A,title='Label',description='Name of the argument in UI');value=Field(default=_A,title='Value',description='Default value of the argument');minimum=Field(default=_A,title=_N,description='Minimum allowed value for the argumentin UI');maximum=Field(default=_A,title=_N,description='Maximum allowed value for the argumentin UI');step=Field(default=_A,title=_N,description='Step for changing value of the argumentin UI');choices=Field(default=_A,title='Choices',description='Possible values for the argument')
class ScriptInfo(BaseModel):name=Field(default=_A,title=_F,description='Script name');is_alwayson=Field(default=_A,title='IsAlwayson',description='Flag specifying whether this script is an alwayson script');is_img2img=Field(default=_A,title='IsImg2img',description='Flag specifying whether this script is an img2img script');args=Field(title='Arguments',description="List of script's arguments")