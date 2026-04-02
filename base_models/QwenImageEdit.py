from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
#from nunchaku import NunchakuQwenImageTransformer2DModel

class QwenImageEdit:
	
	def model_name (self) -> str:
		return 'Qwen/Qwen-Image-Edit'
	
	def pipeline_class (self):
		return QwenImageEditPlusPipeline
	
	def transformer_class (self):
		return QwenImageTransformer2DModel
	
	def lightning_lora (self) -> str:
		return 'lightx2v/Qwen-Image-Lightning:Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors'
	
	#def nunchaku_transformer_class (self):
	#	return NunchakuQwenImageTransformer2DModel
	
	def nunchaku_transformer (self) -> str:
		return 'nunchaku-ai/nunchaku-qwen-image-edit/svdq-fp4_r32-qwen-image-edit-lightningv1.0-4steps.safetensors'