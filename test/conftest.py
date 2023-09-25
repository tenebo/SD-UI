_A='session'
import os,pytest,base64
test_files_path=os.path.dirname(__file__)+'/test_files'
def file_to_base64(filename):
	with open(filename,'rb')as A:B=A.read()
	C=str(base64.b64encode(B),'utf-8');return'data:image/png;base64,'+C
@pytest.fixture(scope=_A)
def img2img_basic_image_base64():return file_to_base64(os.path.join(test_files_path,'img2img_basic.png'))
@pytest.fixture(scope=_A)
def mask_basic_image_base64():return file_to_base64(os.path.join(test_files_path,'mask_basic.png'))