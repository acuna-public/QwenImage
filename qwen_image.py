import argparse
import array
import os
import random
import re
from datetime import datetime

import torch
from PIL import Image

from base_models.SDXLTurbo import SDXLTurbo
from base_models.QwenImage import QwenImage
from base_models.QwenImageEdit import QwenImageEdit

from pipe_loader import PipeLoader
from qwen_image_edit import Image2Image
from qwen_image_generate import Text2Image

class Core:
	
	args = None
	parser = None
	width = []
	height = []
	num_inference_steps = None
	true_cfg_scale = None
	torch_dtype = None
	prompts = []
	sizes = []
	pipe = None
	allowed_models = None
	device = None
	
	aspect_ratios = {
		
		'1:1': (1024, 1024),
		'2:3': (832, 1216),
		
	}
	
	models = {
		
		'QwenImage': QwenImage,
		'QwenImageEdit': QwenImageEdit,
		'SDXLTurbo': SDXLTurbo,
		
	}
	
	def __init__ (self, **kwargs):
		
		self.parser = argparse.ArgumentParser ()
		self.kwargs = kwargs
		
		self.allowed_models = self.models.keys ()
		
		self.parser.add_argument (
			'--base-model', '-b',
			type = str,
			default = kwargs.get ('base_model', None),
			choices = self.allowed_models,
			help = 'Base model (allowed values: ' + ', '.join (self.allowed_models) + ')',
		)
		
		self.parser.add_argument (
			'--model', '-m',
			type = str,
			default = kwargs.get ('model', None),
			help = 'Qwen Image model (local path or HuggingFace model id). If not set - it will be downloaded automatically.',
		)
		
		self.parser.add_argument (
			'--image',
			action = 'append',
			default = kwargs.get ('image', []),
			help = 'Images folder of image file. Can be multiple.',
		)
		
		self.parser.add_argument (
			'--prompt', '-p',
			action = 'append',
			default = kwargs.get ('prompt', []),
			help = 'Prompt. Can be multiple. If not set - folder with images and prompts in txt files with same names as every image needed in main folder.',
		)
		
		self.parser.add_argument (
			'--negative-prompt', '-n',
			default = kwargs.get ('negative_prompt', ' '),
			help = 'Negative prompt (optional)',
		)
		
		self.parser.add_argument (
			'--positive-magic',
			default = kwargs.get ('positive_magic', 'Ultra HD, 4K, cinematic composition'),
			help = 'Positive magic tags for more realism. Appends to every prompt',
		)
		
		self.parser.add_argument (
			'--lora', '-l',
			action = 'append',
			default = kwargs.get ('lora', []),
			help = 'LoRA in format path_or_hf_model_id:weights_file:strength[:trigger]. Can be multiple.',
		)
		
		self.parser.add_argument (
			'--width',
			action = 'append',
			default = kwargs.get ('width', []),
			help = 'Generated image width (1024 by default). Ignored when --ratio is set.',
		)
		
		self.parser.add_argument (
			'--height',
			action = 'append',
			default = kwargs.get ('height', []),
			help = 'Generated image height (1024 by default). Ignored when --ratio is set.',
		)
		
		self.parser.add_argument (
			'--ratio',
			action = 'append',
			default = kwargs.get ('ratio', []),
			help = 'Aspect ratio as string (optional). Ignored when --width and --height are set.',
		)
		
		self.parser.add_argument (
			'--vae',
			type = str,
			default = kwargs.get ('vae', ''),
			help = 'VAE local file',
		)
		
		self.parser.add_argument (
			'--gguf',
			type = str,
			default = kwargs.get ('gguf', ''),
			help = 'GGUF local file',
		)
		
		self.parser.add_argument (
			'--lightning-lora',
			type = str,
			default = kwargs.get ('lightning_lora', None),
			help = 'Lightning LoRA (local path or HuggingFace model id) for reduce inference steps number and increase details (recommended)',
		)
		
		self.parser.add_argument (
			'--nunchaku-transformer',
			type = str,
			default = kwargs.get ('nunchaku_transformer', None),
			help = 'Nunchaku transformer to reduce CPU memory',
		)
		
		self.parser.add_argument (
			'--wildcards-path',
			type = str,
			default = kwargs.get ('wildcards_path', os.getcwd ()),
			help = 'Path to wildcards folder',
		)
		
		self.parser.add_argument (
			'--seed', '-s',
			type = int,
			default = kwargs.get ('seed', 0),
			help = 'Seed (random by default)',
		)
		
		self.parser.add_argument (
			'--scheduler',
			type = str,
			default = kwargs.get ('scheduler', 'Euler a'),
			help = 'Scheduler name (Euler a by default)',
		)
		
		self.parser.add_argument (
			'--cfg-scale',
			type = float,
			default = kwargs.get ('cfg_scale', None),
			help = 'CFG scale (LoRA according by default)',
		)
		
		self.parser.add_argument (
			'--steps',
			type = int,
			default = kwargs.get ('steps', None),
			help = 'Inference steps number (LoRA according by default)',
		)
		
		self.parser.add_argument (
			'--guidance-scale',
			type = float,
			default = kwargs.get ('guidance_scale', 1.0),
			help = 'Guidance scale (1.0 by default)',
		)
		
		self.parser.add_argument (
			'--images-per-prompt',
			type = int,
			default = kwargs.get ('images_per_prompt', 1),
			help = 'Images number per prompt (1 by default)',
		)
		
		self.parser.add_argument (
			'--output-path',
			type = str,
			default = kwargs.get ('output_path', os.getcwd ()),
			help = 'Images output path (current dir by default)',
		)
		
		self.parser.add_argument (
			'--hf-token',
			type = str,
			default = kwargs.get ('hf_token', None),
			help = 'HuggingFace token. Set it if models downloading is slow or stuck.',
		)
		
		self.parser.add_argument (
			'--debug', '-d',
			type = int,
			default = kwargs.get ('debug', 0),
			help = 'Debug mode'
		)
		
		self.parser.add_argument (
			'--show-prompt',
			action = 'store_true',
			default = kwargs.get ('show_prompt', False),
			help = 'Only show prompt (For wildcards generation results check, etc.)'
		)
		
		self.parser.add_argument (
			'--verbose', '-v',
			action = 'store_false',
			default = kwargs.get ('verbose', False),
			help = 'Verbose mode'
		)
	
	def load (self):
		
		print ('Parsing arguments...')
		
		self.args = vars (self.parser.parse_args (
			# args = None if sys.argv[1:] or self.kwargs else ['--help']
		))
		
		if self.args['show_prompt'] is False:
			
			if self.args['base_model'] in self.models:
				
				if self.args['ratio']:
					
					if type (self.args['ratio']) is not array:
						self.args['ratio'] = [self.args['ratio']]
					
					for i in range (len (self.args['ratio'])):
						
						if self.args['ratio'][i] in self.aspect_ratios:
							
							ratio = self.aspect_ratios[self.args['ratio'][i]]
							
							self.width.append (ratio[0])
							self.height.append (ratio[1])
							
						else:
							
							sides = self.args['ratio'][i].split (':')
							
							if sides[1] + ':' + sides[0] in self.aspect_ratios:
								
								ratio = self.aspect_ratios[sides[1] + ':' + sides[0]]
								
								self.width.append (ratio[0])
								self.height.append (ratio[1])
							
							else:
								
								aspect_ratios = []
								
								for key, value in self.aspect_ratios:
									
									sides = key.split (':')
									
									if sides[0] != '1' and sides[1] != '1':
										
										aspect_ratios.append (sides[0] + ':' + sides[1])
										aspect_ratios.append (sides[1] + ':' + sides[0])
										
									aspect_ratios.append (key)
								
								raise ValueError ('Wrong aspect ratio ' + str (self.args['ratio'][i]) + '. Allowed aspect ratios: ' + ', '.join (aspect_ratios) + '.')
								
				elif self.args['width'] or self.args['height']:
					
					if self.args['width'] and self.args['height']:
						
						if type (self.args['width']) is not array:
							self.args['width'] = [self.args['width']]
						
						if type (self.args['height']) is not array:
							self.args['height'] = [self.args['height']]
						
						for i in range (len (self.args['width'])):
							
							self.width.append (self.args['width'][i])
							self.height.append (self.args['height'][i])
						
					else:
						raise ValueError ('You need to set width and height both.')
					
				else:
					
					self.width.append (1024)
					self.height.append (1024)
					
				model = self.models[self.args['base_model']] ()
				
				if self.args['model'] is None:
					self.args['model'] = model.model_name ()
				
				if self.args['lightning_lora'] is None:
					self.args['lightning_lora'] = model.lightning_lora ()
				
				if self.args['nunchaku_transformer'] is None:
					self.args['nunchaku_transformer'] = model.nunchaku_transformer ()
				
				if self.args['hf_token'] is None:
					if 'HF_TOKEN' in os.environ:
						self.args['hf_token'] = os.environ['HF_TOKEN']
				elif self.args['hf_token'] != '':
					os.environ['HF_TOKEN'] = self.args['hf_token']
				
				if torch.cuda.is_available ():
					self.torch_dtype = torch.bfloat16
					self.device = 'cuda'
				else:
					self.torch_dtype = torch.bfloat16
					self.device = 'cpu'
				
				if self.args['cfg_scale'] is None:
					self.true_cfg_scale = 4.0 if self.args['lightning_lora'] == '' else 1.0
				else:
					self.true_cfg_scale = self.args['cfg_scale']
				
				if self.args['steps'] is None:
					self.num_inference_steps = 50 if self.args['lightning_lora'] == '' else 4
				else:
					self.num_inference_steps = self.args['steps']
				
				print ('Load pipeline...')
				
				if self.args['debug'] == 0:
					self.pipe = PipeLoader (self, model).init ()
				
				if len (self.args['image']) > 0:
					image_class = Image2Image (self)
				else:
					image_class = Text2Image (self)
				
				image_class.process ()
				
			else:
				raise ValueError ('Wrong base model ' + str (self.args['base_model']) + '. Allowed models: ' + ', '.join (self.allowed_models) + '.')
			
		else:
			
			self.prompts = self.args['prompt']
			
			for prompt in self.prep_prompts ():
				print (self.process_prompt (prompt))
			
	def load_image (self, path):
		return Image.open (path).convert ('RGB')
	
	def save_images (self, pipes, path, name, extension):
		
		i = 0
		
		for pipe in pipes:
			
			i += 1
			
			i2 = 0
			
			for image in pipe['images']:
				
				i2 += 1
				
				file = os.path.abspath (os.path.join (path, name + '_' + str (i) + '_' + str (i2) + extension))
				
				if self.args['debug'] == 0:
					image.save (file)
				
				print (file, 'generated successfully')
	
	def process_prompt (self, prompt) -> str:
		
		# Wildcards
		
		pat = re.compile (r'__(.+?)__')
		
		pos = 0
		
		while m := pat.search (prompt, pos):
			
			pos = m.start () + 1
			
			file = os.path.abspath (os.path.join (self.args['wildcards_path'], m[1] + '.txt'))
			
			if os.path.exists (file):
				line = self.random_line (file)
				prompt = prompt.replace (m[0], line, 1)
			else:
				raise ValueError ('Wildcard file ' + file + ' not found')
			
		# OR clause
		
		pattern = re.compile (r'{(.+?)}')
		
		pos = 0
		
		while m := pattern.search (prompt, pos):
			
			pos = m.start () + 1
			
			text = str (m[1]).split ('|')
			
			line = text[random.randrange (0, len (text))]
			prompt = self.process_prompt (prompt.replace (m[0], line, 1))
		
		return prompt
	
	def random_line (self, file) -> str:
		
		file = open (file, 'rb')
		text = []
		
		for line in file:
			text.append (line.strip ())
		
		file.close ()
		
		return text[random.randrange (0, len (text))].decode ('utf-8')
	
	def get_date (self) -> str:
		return datetime.today ().strftime ('%Y%m%d%H%M%S%f')
	
	def get_prompts (self, prompt_file):
		
		prompts = []
		
		file = open (prompt_file, 'rb')
		
		self.verbose ('Reading prompts file: ' + prompt_file)
		
		for line in file:
			prompts.append (line.strip ().decode ('utf-8'))
		
		return prompts
		
	def prep_prompts (self):
		
		if type (self.prompts) is not array:
			self.prompts = [self.prompts]
		
		return self.prompts
	
	def progress (self, step, timestep, latents):
		print (step, timestep, latents[0][0][0][0])
	
	def verbose (self, mess):
		
		if self.args['verbose']:
			print (mess)