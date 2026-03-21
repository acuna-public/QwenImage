import os

import torch
from PIL import Image
from diffusers import DiffusionPipeline

class QwenImageGenerate:
	
	pipe = None
	width = None
	height = None
	
	def __init__ (self, core):
		
		self.core = core
		
		if self.core.args['model'] is None:
			self.core.args['model'] = 'Qwen/Qwen-Image'
		
		if self.core.args['lightning_lora'] is None:
			self.core.args['lightning_lora'] = 'lightx2v/Qwen-Image-2512-Lightning:Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors'
		
		if self.core.args['ratio'] is not None:
			self.width = self.aspect_ratios[self.core.args['ratio']][0]
			self.height = self.aspect_ratios[self.core.args['ratio']][1]
		else:
			self.width = self.core.args['width']
			self.height = self.core.args['height']
		
		self.pipe = self.core.pipe_init (DiffusionPipeline)
	
	def process_images (self):
		
		prompt = self.core.args['prompt']
		
		if self.core.args['positive_magic'] != '':
			prompt += ', ' + self.core.args['positive_magic']
		
		with torch.inference_mode ():
			output = self.pipe ({
				
				'prompt': prompt,
				'generator': torch.Generator (device = self.core.device).manual_seed (self.core.args['seed']),
				'true_cfg_scale': self.core.true_cfg_scale,
				'negative_prompt': self.core.args['negative_prompt'],
				'num_inference_steps': self.core.num_inference_steps,
				'guidance_scale': self.core.args['guidance_scale'],
				'num_images_per_prompt': self.core.args['images_per_prompt'],
				'width': self.width,
				'height': self.height,
				
			})
		
		return output
	
	def load_image (self, path):
		return Image.open (path).convert ('RGB')
	
	def process (self):
		
		output = self.process_images ()
		
		file = os.path.join (os.getcwd (), self.core.args['output_name'])
		
		output.images[0].save (file)
		
		print (file, ' generated successfully')