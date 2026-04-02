from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
#from nunchaku import NunchakuQwenImageTransformer2DModel

class QwenImage:
	
	def model_name (self) -> str:
		return 'Qwen/Qwen-Image'
	
	def pipeline_class (self):
		return QwenImagePipeline
	
	def transformer_class (self):
		return QwenImageTransformer2DModel
	
	def lightning_lora (self) -> str:
		return 'lightx2v/Qwen-Image-Lightning:Qwen-Image-Lightning-4steps-V2.0.safetensors'
	
	#def nunchaku_transformer_class (self):
	#	return NunchakuQwenImageTransformer2DModel
	
	def nunchaku_transformer (self) -> str:
		return 'nunchaku-ai/nunchaku-qwen-image/svdq-fp4_r32-qwen-image-lightningv1.0-4steps.safetensors'