import torch

class QwenImageGenerate:
	
	pipe = None
	
	def __init__ (self, core, **kwargs):
		
		self.core = core
		
		if self.core.args['model'] is None:
			self.core.args['model'] = 'Qwen/Qwen-Image'
		
		if self.core.args['lightning_lora'] is None:
			self.core.args['lightning_lora'] = 'lightx2v/Qwen-Image-2512-Lightning:Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors'
		
		if self.core.args['nunchaku_transformer'] is None:
			self.core.args['nunchaku_transformer'] = 'nunchaku-ai/nunchaku-qwen-image/svdq-fp4_r32-qwen-image-lightningv1.0-4steps.safetensors'
		
		if self.core.args['debug'] == 0:
			self.pipe = self.core.pipe_init (kwargs.pop ('pipeline_class'))
	
	def process_images (self):
		
		pipes = {}
		
		self.core.sizes = []
		
		for i in range (0, len (self.core.width)):
			
			width = self.core.width[i]
			
			if width not in pipes:
				pipes[width] = []
			
			pipe = pipes[width]
			
			self.core.sizes.append ([self.core.width[i], self.core.height[i]])
			
			for prompt in self.core.prompts:
				
				if self.core.args['positive_magic'] != '':
					prompt += ', ' + self.core.args['positive_magic']
				
				if self.core.args['debug'] == 0:
					
					with torch.inference_mode ():
						pipe.append (self.pipe ({
							
							'width': self.core.width[i],
							'height': self.core.height[i],
							'prompt': self.core.process_prompt (prompt),
							'negative_prompt': self.core.args['negative_prompt'],
							'true_cfg_scale': self.core.true_cfg_scale,
							'num_inference_steps': self.core.num_inference_steps,
							'guidance_scale': self.core.args['guidance_scale'],
							'num_images_per_prompt': self.core.args['images_per_prompt'],
							'generator': torch.Generator (device = self.core.device).manual_seed (self.core.args['seed']),
							
						}))
					
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
			
			self.core.save_images (pipes[width], self.core.args['output_path'], self.core.get_date () + '_' + str (sizes[0]) + '_' + str (sizes[1]), '.jpg')