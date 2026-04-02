import torch

class Text2Image:
	
	def __init__ (self, core):
		
		self.core = core
	
	def process_images (self):
		
		pipes = { }
		
		self.core.sizes = []
		
		for i in range (len (self.core.width)):
			
			width = self.core.width[i]
			
			if width not in pipes:
				pipes[width] = []
			
			pipe = pipes[width]
			
			self.core.sizes.append ([self.core.width[i], self.core.height[i]])
			
			for prompt in self.core.prep_prompts ():
				
				if self.core.args['positive_magic'] != '':
					prompt += ', ' + self.core.args['positive_magic']
				
				if self.core.args['debug'] == 0:
					
					with torch.autocast (self.core.device), torch.inference_mode ():
						
						pipe.append (self.core.pipe (
							
							width = self.core.width[i],
							height = self.core.height[i],
							prompt = self.core.process_prompt (prompt),
							negative_prompt = self.core.args['negative_prompt'],
							true_cfg_scale = self.core.true_cfg_scale,
							num_inference_steps = self.core.num_inference_steps,
							guidance_scale = self.core.args['guidance_scale'],
							num_images_per_prompt = self.core.args['images_per_prompt'],
							generator = torch.Generator (
								device = self.core.device
							).manual_seed (self.core.args['seed']),
							
						))
			
				else:
					pipe.append ({ 'images': [0, 1] })
			
			pipes[width] = pipe
		
		return pipes
	
	def process (self):
		
		self.core.prompts = self.core.args['prompt']
		
		pipes = self.process_images ()
		
		i = 0
		
		for width in pipes:
			
			sizes = self.core.sizes[i]
			
			i += 1
			
			self.core.save_images (
				pipes[width],
				self.core.args['output_path'],
				self.core.get_date () + '_' + str (sizes[0]) + '_' + str (sizes[1]), '.jpg'
			)