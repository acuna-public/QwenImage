import os
import pathlib

import torch
from diffusers.utils import load_image

class QwenImageEdit:
	
	pipe = None
	prompts = []
	
	def __init__ (self, core, **kwargs):
		
		self.core = core
		
		if self.core.args['model'] is None:
			self.core.args['model'] = 'Qwen/Qwen-Image-Edit'
		
		if self.core.args['lightning_lora'] is None:
			self.core.args['lightning_lora'] = 'lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
		
		if self.core.args['debug'] == 0:
			self.pipe = self.core.pipe_init (kwargs.pop ('pipeline_class'))
		
	def process_images (self, images):
		
		pipes = []
		
		for prompt in self.prompts:
			
			if self.core.args['debug'] == 0:
				
				with torch.inference_mode ():
					pipes.append (self.pipe ({
						
						'image': images,
						'prompt': self.core.process_prompt (prompt),
						'generator': torch.Generator (device = self.core.device).manual_seed (self.core.args['seed']),
						'true_cfg_scale': self.core.true_cfg_scale,
						'negative_prompt': self.core.args['negative_prompt'],
						'num_inference_steps': self.core.num_inference_steps,
						'guidance_scale': self.core.args['guidance_scale'],
						'num_images_per_prompt': self.core.args['images_per_prompt'],
						
					}))
			else:
				pipes.append ({'images': [0, 1]})
		
		return pipes
	
	def process (self):
		
		if os.path.isdir (self.core.args['image'][0]):
			
			for (root, directories, files) in os.walk (self.core.args['image'][0]):
				
				for filename in files:
					
					prompt_file = os.path.abspath (os.path.join (root, filename + '.txt'))
					
					if len (self.core.args['prompt']) > 0 or os.path.exists (prompt_file):
						
						root_file = os.path.join (root, filename)
						stem = pathlib.Path (filename).stem
						extension = os.path.splitext (filename)[1]
						
						self.prompts = self.core.get_prompts (prompt_file)
						
						if len (self.prompts) > 0:
							
							if len (self.core.args['image']) > 1:
								
								for i in range (1, len (self.core.args['image'])):
									
									for (root2, directories2, files2) in os.walk (self.core.args['image'][i]):
										
										for filename2 in files2:
											
											pipes = self.process_images ([
												self.core.load_image (root_file),
												self.core.load_image (os.path.join (root2, filename2)),
											])
											
											folder = os.path.basename (root2)
											stem2 = pathlib.Path (filename2).stem
											
											folder_path = os.path.abspath (os.path.join (self.core.args['output_path'], folder))
											
											if not os.path.exists (folder_path):
												os.makedirs (folder_path)
											
											self.core.save_images (pipes, folder_path, stem + '_' + stem2, extension)
										
										break
										
							else:
								
								pipes = self.process_images ([
									self.core.load_image (root_file)
								])
								
								self.core.save_images (pipes, self.core.args['output_path'], self.core.get_date (), '.jpg')
				
				break
				
		else:
			
			images = []
			
			for image in self.core.args['image']:
				images.append (load_image (image))
			
			for prompt in self.core.args['prompt']:
				self.prompts.append (prompt)
			
			pipes = self.process_images (images)
			self.core.save_images (pipes, self.core.args['output_path'], self.core.get_date (), '.jpg')