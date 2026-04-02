from optimum.intel import OVStableDiffusionXLPipeline

class SDXLTurbo:
	
	def model_name (self) -> str:
		return 'sdxl-turbo-openvino-int8'
	
	def pipeline_class (self):
		return OVStableDiffusionXLPipeline
	
	def transformer_class (self):
		return None
	
	def lightning_lora (self) -> str:
		return ''
	
	def nunchaku_transformer_class (self):
		return None
	
	def nunchaku_transformer (self) -> str:
		return ''