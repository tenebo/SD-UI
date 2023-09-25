_I='tiling'
_H='sampler_index'
_G='restore_faces'
_F='prompt'
_E='negative_prompt'
_D='n_iter'
_C='height'
_B='enable_hr'
_A='batch_size'
import pytest,requests
@pytest.fixture()
def url_txt2img(base_url):return f"{base_url}/sdapi/v1/txt2img"
@pytest.fixture()
def simple_txt2img_request():A=False;return{_A:1,'cfg_scale':7,'denoising_strength':0,_B:A,'eta':0,'firstphase_height':0,'firstphase_width':0,_C:64,_D:1,_E:'',_F:'example prompt',_G:A,'s_churn':0,'s_noise':1,'s_tmax':0,'s_tmin':0,_H:'Euler a','seed':-1,'seed_resize_from_h':-1,'seed_resize_from_w':-1,'steps':3,'styles':[],'subseed':-1,'subseed_strength':0,_I:A,'width':64}
def test_txt2img_simple_performed(url_txt2img,simple_txt2img_request):assert requests.post(url_txt2img,json=simple_txt2img_request).status_code==200
def test_txt2img_with_negative_prompt_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_E]='example negative prompt';assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_with_complex_prompt_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_F]='((emphasis)), (emphasis1:1.1), [to:1], [from::2], [from:to:0.3], [alt|alt1]';assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_not_square_image_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_C]=128;assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_with_hrfix_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_B]=True;assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_with_tiling_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_I]=True;assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_with_restore_faces_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_G]=True;assert requests.post(url_txt2img,json=A).status_code==200
@pytest.mark.parametrize('sampler',['PLMS','DDIM','UniPC'])
def test_txt2img_with_vanilla_sampler_performed(url_txt2img,simple_txt2img_request,sampler):A=simple_txt2img_request;A[_H]=sampler;assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_multiple_batches_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_D]=2;assert requests.post(url_txt2img,json=A).status_code==200
def test_txt2img_batch_performed(url_txt2img,simple_txt2img_request):A=simple_txt2img_request;A[_A]=2;assert requests.post(url_txt2img,json=A).status_code==200