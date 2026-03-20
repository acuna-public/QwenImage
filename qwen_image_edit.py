import os
import pathlib

import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

class QwenImageEdit:
	
	pipe = None
	prompt = None
	
	def __init__ (self, core):
		
		self.core = core
		
		if self.core.args['model'] is None:
			self.core.args['model'] = 'Qwen/Qwen-Image-Edit'
		
		if self.core.args['lightning_lora'] is None:
			self.core.args['lightning_lora'] = 'lightx2v/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'
		
		self.pipe = self.core.pipe_init (QwenImageEditPlusPipeline)
	
	def process_images (self, images):
		
		with torch.inference_mode ():
			output = self.pipe ({
				
				'image': images,
				'prompt': self.prompt,
				'generator': torch.Generator (device = self.core.device).manual_seed (self.core.args['seed']),
				'true_cfg_scale': self.core.true_cfg_scale,
				'negative_prompt': self.core.args['negative_prompt'],
				'num_inference_steps': self.core.num_inference_steps,
				'guidance_scale': self.core.args['guidance_scale'],
				'num_images_per_prompt': self.core.args['images_per_prompt'],
				
			})
		
		return output
	
	def process (self):
		
		if os.path.isdir (self.core.args['image'][0]):
			
			for (root, directories, files) in os.walk (self.core.args['image'][0]):
				
				for filename in files:
					
					prompt_file = os.path.abspath (os.path.join (root, filename + '.txt'))
					
					if self.core.args['prompt'] != '' or os.path.exists (prompt_file):
						
						root_file = os.path.join (root, filename)
						stem = pathlib.Path (filename).stem
						extension = os.path.splitext (filename)[1]
						
						if self.core.args['prompt'] != '':
							self.prompt = self.core.args['prompt']
						else:
							with open (prompt_file) as f:
								self.prompt = f.read ()
						
						if self.prompt != '':
							
							if len (self.core.args['image']) > 1:
								
								for i in range (1, len (self.core.args['image'])):
									
									for (root2, directories2, files2) in os.walk (self.core.args['image'][i]):
										
										for filename2 in files2:
											
											output = self.process_images ([
												self.core.load_image (root_file),
												self.core.load_image (os.path.join (root2, filename2)),
											])
											
											folder = os.path.basename (root2)
											stem2 = pathlib.Path (filename2).stem
											
											folder_path = os.path.join (os.getcwd (), folder)
											
											if not os.path.exists (folder_path):
												os.makedirs (folder_path)
											
											file = os.path.join (folder_path, stem + '_' + stem2 + extension)
											
											output.images[0].save (file)
											
											print (file, ' generated successfully')
							
							else:
								
								output = self.process_images ([
									self.core.load_image (root_file)
								])
								
								file = os.path.join (os.getcwd (), filename)
								
								output.images[0].save (file)
								
								print (file, ' generated successfully')
		
		else:
			
			images = []
			
			for i in range (0, len (self.core.args['image'])):
				images.append (load_image (self.core.args['image'][i]))
			
			output = self.process_images (images)
			
			file = os.path.join (os.getcwd (), self.core.args['output_name'])
			
			output.images[0].save (file)
			
			print (file, ' generated successfully')