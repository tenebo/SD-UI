from modules import shared
class FaceRestoration:
	def name(A):return'None'
	def restore(A,np_image):return np_image
def restore_faces(np_image):
	A=np_image;B=[A for A in shared.face_restorers if A.name()==shared.opts.face_restoration_model or shared.opts.face_restoration_model is None]
	if len(B)==0:return A
	C=B[0];return C.restore(A)