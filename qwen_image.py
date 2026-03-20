import argparse
import math
import sys

import torch
from diffusers import QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler

from qwen_image_edit import QwenImageEdit
from qwen_image_generate import QwenImageGenerate

#from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
#from nunchaku.utils import get_gpu_memory, get_precision

class QwenImage:
	
	args = None
	parser = None
	num_inference_steps = None
	true_cfg_scale = None
	torch_dtype = None
	device = None
	
	def __init__ (self):
		
		self.parser = argparse.ArgumentParser ()
		
		self.parser.add_argument (
			'--model', '-m',
			type = str,
			default = None,
			help = 'Full path to Qwen Image model. If not set - it will be downloaded automatically.',
		)
		
		self.parser.add_argument (
			'--width',
			type = int,
			default = 1024,
			help = 'Generated image width (1024 by default). Ignored when --ratio is set.',
		)
		
		self.parser.add_argument (
			'--height',
			type = int,
			default = 1024,
			help = 'Generated image height (1024 by default). Ignored when --ratio is set.',
		)
		
		self.parser.add_argument (
			'--ratio',
			type = str,
			default = None,
			help = 'Aspect ratio as string (optional). Ignored when --width and --height are set.',
		)
		
		self.parser.add_argument (
			'--lightning-lora',
			default = None,
			type = str,
			help = 'Lightning lora (local path or Huggingface model id) for reduce inference steps number and increase details (recommended)',
		)
		
		self.parser.add_argument (
			'--lightning-lora-weights',
			type = str,
			default = None,
			help = 'Lightning lora weights file name (optional)',
		)
		
		self.parser.add_argument (
			'--image',
			action = 'append',
			default = [],
			help = 'Images folder of image file (can be multiple)',
		)
		
		self.parser.add_argument (
			'--prompt', '-p',
			default = '',
			help = 'Prompt (if not set - folder with images and prompts in txt files with same names as every image needed in main folder)',
		)
		
		self.parser.add_argument (
			'--negative-prompt', '-n',
			default = ' ',
			help = 'Negative prompt (optional)',
		)
		
		self.parser.add_argument (
			'--positive-magic',
			default = 'Ultra HD, 4K, cinematic composition',
			help = 'Positive magic tags for more realism',
		)
		
		self.parser.add_argument (
			'--seed', '-s',
			type = int,
			default = 0,
			help = 'Seed (random by default)',
		)
		
		self.parser.add_argument (
			'--hf-token',
			type = str,
			default = None,
			help = 'Huggingface token. Set it if models downloading is slow or stuck.',
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
			'--output-name',
			default = '',
			help = 'Result image name if --image is file',
		)
		
		self.parser.add_argument (
			'--verbose', '-v',
			action = 'store_false',
			default = False,
			help = 'Verbose mode'
		)
	
	def load (self):
		
		self.args = vars (self.parser.parse_args (
			args = None if sys.argv[1:] else ['--help']
		))
		
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
		
		if len (self.args['image']) > 0:
			image_class = QwenImageEdit (self)
		else:
			image_class = QwenImageGenerate (self)
		
		image_class.process ()
	
	def pipe_init (self, pipe_cls):
		
		if self.args['lightning_lora'] != '':
			
			transformer = QwenImageTransformer2DModel.from_pretrained (
				self.args['model'],
				subfolder = 'transformer',
				use_safetensors = True,
				torch_dtype = self.torch_dtype,
				token = self.args['hf_token'],
			)
			
			# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/main/generate_with_diffusers.py
			
			scheduler = FlowMatchEulerDiscreteScheduler.from_config ({
				
				'base_image_seq_len': 256,
				'base_shift': math.log (3),  # We use shift=3 in distillation
				'invert_sigmas': False,
				'max_image_seq_len': 8192,
				'max_shift': math.log (3),  # We use shift=3 in distillation
				'num_train_timesteps': 1000,
				'shift': 1.0,
				'shift_terminal': None,  # set shift_terminal to None
				'stochastic_sampling': False,
				'time_shift_type': 'exponential',
				'use_beta_sigmas': False,
				'use_dynamic_shifting': True,
				'use_exponential_sigmas': False,
				'use_karras_sigmas': False,
				
			})
			
			pipe = pipe_cls.from_pretrained (
				self.args['model'],
				transformer = transformer,
				scheduler = scheduler,
				use_safetensors = True,
				torch_dtype = self.torch_dtype,
				token = self.args['hf_token'],
			)
			
			pipe.load_lora_weights (
				self.args['lightning_lora'],
				weight_name = self.args['lightning_lora_weights'],
			)
			
		else:
			pipe = pipe_cls.from_pretrained (
				self.args['model'],
				torch_dtype = self.torch_dtype,
				use_safetensors = True,
				token = self.args['hf_token'],
			)
		
		'''
		elif self.args['nunchaku_transformer'] is not None:

			transformer = NunchakuQwenImageTransformer2DModel.from_pretrained (self.args['nunchaku_transformer'])

			pipe = pipe_cls.from_pretrained (
				self.args['model'],
				torch_dtype = self.torch_dtype,
			)

			if get_gpu_memory () <= 18:

				# use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
				transformer.set_offload (
					True,
					use_pin_memory = False,
					num_blocks_on_gpu = 1,
				)  # increase num_blocks_on_gpu if you have more VRAM

				pipe.enable_model_cpu_offload ()
				pipe._exclude_from_cpu_offload.append ('transformer')

			else:
				pipe.enable_sequential_cpu_offload ()
		'''
		pipe = pipe.to (self.device)
		
		pipe.set_progress_bar_config (disable = None)
		
		return pipe

QwenImage ().load ()