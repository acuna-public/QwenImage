import os
import pathlib

import torch
from diffusers.utils import load_image

class Image2Image:
	
	pipe = None
	
	def __init__ (self, core):
		
		self.core = core
	
	def process_images (self, images):
		
		pipes = []
		
		for prompt in self.core.prep_prompts ():
			
			if self.core.args['debug'] == 0:
				
				with torch.inference_mode ():
					pipes.append (self.core.pipe (
						
						image = images,
						prompt = self.core.process_prompt (prompt),
						generator = torch.Generator (
							device = self.core.device
						).manual_seed (self.core.args['seed']),
						true_cfg_scale = self.core.true_cfg_scale,
						negative_prompt = self.core.args['negative_prompt'],
						num_inference_steps = self.core.num_inference_steps,
						guidance_scale = self.core.args['guidance_scale'],
						num_images_per_prompt = self.core.args['images_per_prompt'],
						
					))
			else:
				pipes.append ({ 'images': [0, 1] })
		
		return pipes
	
	def process (self):
		
		if os.path.isdir (self.core.args['image'][0]):
			
			for (root, directories, files) in os.walk (self.core.args['image'][0]):
				
				self.core.verbose ('Reading ' + root + ' folder...')
				
				prompt_file = os.path.abspath (os.path.join (root, root + '.txt'))
				
				if os.path.exists (prompt_file):
					
					self.core.verbose ('Reading prompts file: ' + prompt_file)
					root_prompts = self.core.get_prompts (prompt_file)
				
				else:
					root_prompts = []
				
				for filename in files:
					
					self.core.verbose ('Reading ' + os.path.abspath (os.path.join (root, filename)) + ' file...')
					
					prompt_file = os.path.abspath (os.path.join (root, filename + '.txt'))
					
					exists = os.path.exists (prompt_file)
					
					if self.core.args['prompt'] or exists:
						
						root_file = os.path.join (root, filename)
						stem = pathlib.Path (filename).stem
						extension = os.path.splitext (filename)[1]
						
						if exists:
							self.core.prompts = self.core.get_prompts (prompt_file)
						elif root_prompts:
							self.core.prompts = root_prompts
						else:
							self.core.prompts = self.core.args['prompt']
						
						if self.core.prompts:
							
							if len (self.core.args['image']) > 1:
								
								for i in range (1, len (self.core.args['image'])):
									
									for (root2, directories2, files2) in os.walk (self.core.args['image'][i]):
										
										self.core.verbose ('Reading ' + root2 + ' folder...')
										
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
											
											self.core.save_images (
												pipes,
												folder_path, stem + '_' + stem2 + extension
											)
											
										break
							
							else:
								
								pipes = self.process_images ([
									self.core.load_image (root_file)
								])
								
								self.core.save_images (
									pipes,
									self.core.args['output_path'],
									self.core.get_date () + '.jpg'
								)
				
				break
		
		else:
			
			images = []
			
			for image in self.core.args['image']:
				
				self.core.verbose ('Reading ' + os.path.abspath (image) + ' file...')
				images.append (load_image (image))
			
			prompt_file = os.path.abspath (self.core.args['image'][0] + '.txt')
			
			if not self.core.args['prompt'] and os.path.exists (prompt_file):
				
				self.core.verbose ('Reading prompts file: ' + prompt_file)
				self.core.prompts = self.core.get_prompts (prompt_file)
			
			else:
				self.core.prompts = self.core.args['prompt']
			
			pipes = self.process_images (images)
			
			self.core.save_images (
				pipes,
				self.core.args['output_path'],
				self.core.get_date () + '.jpg'
			)