import os

import torch

class QwenImageGenerate:
	
	pipe = None
	width = None
	height = None
	prompts = []
	
	def __init__ (self, core, **kwargs):
		
		self.core = core
		
		if self.core.args['model'] is None:
			self.core.args['model'] = 'Qwen/Qwen-Image'
		
		if self.core.args['lightning_lora'] is None:
			self.core.args['lightning_lora'] = 'lightx2v/Qwen-Image-2512-Lightning:Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors'
		
		if self.core.args['ratio'] is not None:
			self.width = self.core.aspect_ratios[self.core.args['ratio']][0]
			self.height = self.core.aspect_ratios[self.core.args['ratio']][1]
		else:
			self.width = self.core.args['width']
			self.height = self.core.args['height']
		
		if self.core.args['debug'] == 0:
			self.pipe = self.core.pipe_init (kwargs.pop ('pipeline_class'))
	
	def process_images (self):
		
		pipes = []
		
		for prompt in self.prompts:
			
			if self.core.args['positive_magic'] != '':
				prompt += ', ' + self.core.args['positive_magic']
			
			if self.core.args['debug'] == 0:
				
				with torch.inference_mode ():
					pipes.append (self.pipe ({
						
						'prompt': self.core.process_prompt (prompt),
						'generator': torch.Generator (device = self.core.device).manual_seed (self.core.args['seed']),
						'true_cfg_scale': self.core.true_cfg_scale,
						'negative_prompt': self.core.args['negative_prompt'],
						'num_inference_steps': self.core.num_inference_steps,
						'guidance_scale': self.core.args['guidance_scale'],
						'num_images_per_prompt': self.core.args['images_per_prompt'],
						'width': self.width,
						'height': self.height,
						
					}))
			else:
				pipes.append ({ 'images': [0, 1] })
		
		return pipes
	
	def process (self):
		
		for prompt in self.core.args['prompt']:
			self.prompts.append (prompt)
		
		pipes = self.process_images ()
		
		self.core.save_images (pipes, self.core.args['output_path'], self.core.get_date (), '.jpg')