import os

import torch
from diffusers import DiffusionPipeline

class QwenImageGenerate:
	
	pipe = None
	width = None
	height = None
	
	def __init__ (self, core):
		
		self.core = core
		
		self.aspect_ratios = {
			
			'1:1': (1328, 1328),
			'16:9': (1664, 928),
			'9:16': (928, 1664),
			'4:3': (1472, 1104),
			'3:4': (1104, 1472),
			'3:2': (1584, 1056),
			'2:3': (1056, 1584),
			
		}
		
		if self.core.args['model'] is None:
			self.core.args['model'] = 'Qwen/Qwen-Image'
		
		if self.core.args['lightning_lora'] is None:
			self.core.args['lightning_lora'] = 'lightx2v/Qwen-Image-2512-Lightning/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors'
		
		if self.core.args['ratio'] is not None:
			self.width = self.aspect_ratios[self.core.args['ratio']][0]
			self.height = self.aspect_ratios[self.core.args['ratio']][1]
		else:
			self.width = self.core.args['width']
			self.height = self.core.args['height']
		
		self.pipe = self.core.pipe_init (DiffusionPipeline)
	
	def process_images (self):
		
		with torch.inference_mode ():
			output = self.pipe ({
				
				'prompt': self.core.args['prompt'] + ", " + self.core.args['positive_magic'],
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
	
	def process (self):
		
		output = self.process_images ()
		
		file = os.path.join (os.getcwd (), self.core.args['output_name'])
		
		output.images[0].save (file)
		
		print (file, ' generated successfully')