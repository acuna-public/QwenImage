import argparse
import math
import os
import random
import re
import sys
from datetime import datetime

import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, DiffusionPipeline, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, LMSDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel, UniPCMultistepScheduler

from qwen_image_edit import QwenImageEdit
from qwen_image_generate import QwenImageGenerate

from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory

class QwenImage:
	
	args = None
	parser = None
	width = []
	height = []
	num_inference_steps = None
	true_cfg_scale = None
	torch_dtype = None
	prompts = []
	device = None
	
	aspect_ratios = {
		
		'1:1': (1328, 1328),
		'16:9': (1664, 928),
		'9:16': (928, 1664),
		'4:3': (1472, 1104),
		'3:4': (1104, 1472),
		'3:2': (1584, 1056),
		'2:3': (1056, 1584),
		
	}
	
	def __init__ (self):
		
		self.parser = argparse.ArgumentParser ()
		
		self.parser.add_argument (
			'--model', '-m',
			type = str,
			default = None,
			help = 'Qwen Image model (local path or HuggingFace model id). If not set - it will be downloaded automatically.',
		)
		
		self.parser.add_argument (
			'--image',
			action = 'append',
			default = [],
			help = 'Images folder of image file. Can be multiple.',
		)
		
		self.parser.add_argument (
			'--prompt', '-p',
			action = 'append',
			default = [],
			help = 'Prompt. Can be multiple. If not set - folder with images and prompts in txt files with same names as every image needed in main folder.',
		)
		
		self.parser.add_argument (
			'--negative-prompt', '-n',
			default = ' ',
			help = 'Negative prompt (optional)',
		)
		
		self.parser.add_argument (
			'--positive-magic',
			default = 'Ultra HD, 4K, cinematic composition',
			help = 'Positive magic tags for more realism. Appends to every prompt',
		)
		
		self.parser.add_argument (
			'--lora', '-l',
			action = 'append',
			default = [],
			help = 'LoRA in format path_or_hf_model_id:weights_file:strength[:trigger]. Can be multiple.',
		)
		
		self.parser.add_argument (
			'--width',
			action = 'append',
			default = [],
			help = 'Generated image width (1024 by default). Ignored when --ratio is set.',
		)
		
		self.parser.add_argument (
			'--height',
			action = 'append',
			default = [],
			help = 'Generated image height (1024 by default). Ignored when --ratio is set.',
		)
		
		self.parser.add_argument (
			'--ratio',
			action = 'append',
			default = [],
			help = 'Aspect ratio as string (optional). Ignored when --width and --height are set.',
		)
		
		self.parser.add_argument (
			'--gguf',
			default = '',
			type = str,
			help = 'GGUF file path or url',
		)
		
		self.parser.add_argument (
			'--lightning-lora',
			default = None,
			type = str,
			help = 'Lightning LoRA (local path or HuggingFace model id) for reduce inference steps number and increase details (recommended)',
		)
		
		self.parser.add_argument (
			'--nunchaku-transformer',
			default = None,
			type = str,
			help = 'Nunchaku transformer to reduce CPU memory',
		)
		
		self.parser.add_argument (
			'--wildcards-path',
			default = os.getcwd (),
			type = str,
			help = 'Path to wildcards folder',
		)
		
		self.parser.add_argument (
			'--seed', '-s',
			type = int,
			default = 0,
			help = 'Seed (random by default)',
		)
		
		self.parser.add_argument (
			'--scheduler',
			type = str,
			default = 'DPM++ 2M Karras',
			help = 'Scheduler name (DPM++ 2M Karras by default)',
		)
		
		self.parser.add_argument (
			'--cfg-scale',
			type = float,
			default = None,
			help = 'CFG scale (LoRA according by default)',
		)
		
		self.parser.add_argument (
			'--steps',
			type = int,
			default = None,
			help = 'Inference steps number (LoRA according by default)',
		)
		
		self.parser.add_argument (
			'--guidance-scale',
			type = float,
			default = 1.0,
			help = 'Guidance scale (1.0 by default)',
		)
		
		self.parser.add_argument (
			'--images-per-prompt',
			type = int,
			default = 1,
			help = 'Images number per prompt (1 by default)',
		)
		
		self.parser.add_argument (
			'--output-path',
			default = os.getcwd (),
			help = 'Images output path (current dir by default)',
		)
		
		self.parser.add_argument (
			'--hf-token',
			type = str,
			default = None,
			help = 'HuggingFace token. Set it if models downloading is slow or stuck.',
		)
		
		self.parser.add_argument (
			'--debug', '-d',
			default = 0,
			help = 'Debug mode'
		)
		
		self.parser.add_argument (
			'--show-prompt',
			action = 'store_true',
			default = False,
			help = 'Only show prompt (For wildcards generation results check, etc.)'
		)
		
		self.parser.add_argument (
			'--verbose', '-V',
			action = 'store_false',
			default = False,
			help = 'Verbose mode'
		)
	
	def load (self):
		
		self.args = vars (self.parser.parse_args (
			args = None if sys.argv[1:] else ['--help']
		))
		
		if self.args['show_prompt'] is False:
			
			if self.args['hf_token'] is None:
				if 'HF_TOKEN' in os.environ:
					self.args['hf_token'] = os.environ['HF_TOKEN']
			elif self.args['hf_token'] != '':
				os.environ['HF_TOKEN'] = self.args['hf_token']
			
			if torch.cuda.is_available ():
				self.torch_dtype = torch.bfloat16
				self.device = 'cuda'
			else:
				self.torch_dtype = torch.float32
				self.device = 'cpu'
			
			if self.args['cfg_scale'] is None:
				self.true_cfg_scale = 4.0 if self.args['lightning_lora'] == '' else 1.0
			else:
				self.true_cfg_scale = self.args['cfg_scale']
			
			if self.args['steps'] is None:
				self.num_inference_steps = 50 if self.args['lightning_lora'] == '' else 4
			else:
				self.num_inference_steps = self.args['steps']
			
			if len (self.args['ratio']) > 0:
				
				for i in range (0, len (self.args['ratio'])):
					
					ratio = self.aspect_ratios[self.args['ratio'][i]]
					
					self.width.append (ratio[0])
					self.height.append (ratio[1])
			
			elif self.args['width'] and self.args['height']:
				
				for i in range (0, len (self.args['width'])):
					
					self.width.append (self.args['width'][i])
					self.height.append (self.args['height'][i])
			
			else:
				
				self.width.append (1024)
				self.height.append (1024)
			
			if len (self.args['image']) > 0:
				image_class = QwenImageEdit (self, pipeline_class = QwenImageEditPlusPipeline)
			else:
				image_class = QwenImageGenerate (self, pipeline_class = DiffusionPipeline)
			
			image_class.process ()
		
		else:
			for prompt in self.args['prompt']:
				print (self.process_prompt (prompt))
	
	def get_scheduler (self, config):
		
		schedulers_cls = {
			
			'DDIM': DDIMScheduler,
			'DDPM': DDPMScheduler,
			'DPM++ 2M': DPMSolverMultistepScheduler,
			'DPM++ 2M Karras': DPMSolverMultistepScheduler,
			'DPM++ SDE': DPMSolverSinglestepScheduler,
			'DPM++ SDE Karras': DPMSolverSinglestepScheduler,
			'DPM2': KDPM2DiscreteScheduler,
			'DPM2 Karras': KDPM2DiscreteScheduler,
			'DPM2 a': KDPM2AncestralDiscreteScheduler,
			'DPM2 a Karras': KDPM2AncestralDiscreteScheduler,
			'Euler': EulerDiscreteScheduler,
			'Euler a': EulerAncestralDiscreteScheduler,
			'Heun': HeunDiscreteScheduler,
			'LMS': LMSDiscreteScheduler,
			'FlowMatch': FlowMatchEulerDiscreteScheduler,
			
		}
		
		use_sigmas = [
			
			'DPM++ 2M Karras',
			'DPM++ 2M SDE Karras',
			'DPM++ SDE Karras',
			'DPM2 Karras',
			'DPM2 a Karras',
			'LMS Karras',
		
		]
		
		config['use_karras_sigmas'] = True if self.args['scheduler'] in use_sigmas else False
		
		if self.args['scheduler'] in schedulers_cls:
			scheduler = schedulers_cls[self.args['scheduler']].from_config (config)
		elif self.args['scheduler'] == 'UniPC':
			scheduler = UniPCMultistepScheduler.from_pretrained (self.args['model'])
		else:
			
			schedulers_algs = {
				
				'DPM++ 2M SDE': 'sde-dpmsolver++',
				'DPM++ 2M SDE Karras': 'sde-dpmsolver++',
				'DPM++ SDE Karras': 'sde-dpmsolver++',
				
			}
			
			scheduler = DPMSolverMultistepScheduler.from_config (config)
			
			if self.args['scheduler'] in schedulers_algs:
				scheduler.config.algorithm_type = schedulers_algs[self.args['scheduler']]
			else:
				scheduler.config.algorithm_type = self.args['scheduler']
		
		return scheduler
	
	def pipe_init (self, pipe_cls):
		
		if self.args['gguf'] != '':
			
			transformer = QwenImageTransformer2DModel.from_single_file (
				self.args['gguf'],
				subfolder = 'transformer',
				quantization_config = GGUFQuantizationConfig (compute_dtype = self.torch_dtype),
				torch_dtype = self.torch_dtype,
				config = self.args['model'],
			)
			
		elif self.args['nunchaku_transformer'] != '':
			
			transformer = NunchakuQwenImageTransformer2DModel.from_pretrained (self.args['nunchaku_transformer'])
			
			if get_gpu_memory () <= 18:
				
				# use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
				transformer.set_offload (
					True,
					use_pin_memory = False,
					num_blocks_on_gpu = 1,
				)  # increase num_blocks_on_gpu if you have more VRAM
			
		else:
			
			transformer = QwenImageTransformer2DModel.from_pretrained (
				self.args['model'],
				subfolder = 'transformer',
				torch_dtype = self.torch_dtype,
			)
		
		vae = AutoencoderKL.from_pretrained (
			self.args['model'],
			subfolder = 'vae',
			torch_dtype = self.torch_dtype
		)
		
		if self.args['lightning_lora'] != '':
			config = {
				
				'base_image_seq_len': 256,
				'max_image_seq_len': 8192,
				'base_shift': math.log (3),
				'max_shift': math.log (3),
				'invert_sigmas': False,
				'shift': 1.0,
				'shift_terminal': None,
				'stochastic_sampling': False,
				'time_shift_type': 'exponential',
				'use_beta_sigmas': False,
				'use_dynamic_shifting': True,
				'use_exponential_sigmas': False,
				'use_karras_sigmas': False,
				
			}
		else:
			config = { }
		
		pipe = pipe_cls.from_pretrained (
			self.args['model'],
			transformer = transformer,
			vae = vae,
			scheduler = self.get_scheduler (config),
			use_safetensors = True,
			torch_dtype = self.torch_dtype,
			token = self.args['hf_token'],
		)
		
		if get_gpu_memory () <= 18:
			
			pipe.enable_model_cpu_offload ()
			pipe._exclude_from_cpu_offload.append ('transformer')
		
		else:
			pipe.enable_sequential_cpu_offload ()
		
		if self.args['lightning_lora'] != '':
			
			model_id, weights_path = self.args['lightning_lora'].split (':')
			
			pipe.load_lora_weights (
				model_id,
				weights_path = weights_path,
			)
		
		for lora in self.args['lora']:
			
			lora = lora.split (':')
			
			pipe.load_lora_weights (
				lora[0],
				weights_path = lora[1],
				token = lora[3] if len (lora) == 4 else ''
			)
			
			pipe.fuse_lora (lora_scale = lora[2])
			pipe.unload_lora_weights ()
		
		pipe = pipe.to (self.device)
		
		pipe.set_progress_bar_config (disable = None)
		
		return pipe
	
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
	
	def process_prompt (self, prompt):
		
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
		
		pat = re.compile (r'\{(.+?)\}')
		
		pos = 0
		
		while m := pat.search (prompt, pos):
			
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
	
	def verbose (self, mess):
		
		if self.args['verbose']:
			print (mess)

QwenImage ().load ()