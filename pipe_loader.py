import math

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler, GGUFQuantizationConfig, HeunDiscreteScheduler, KDPM2AncestralDiscreteScheduler, KDPM2DiscreteScheduler, LMSDiscreteScheduler, DiffusionPipeline

class PipeLoader:
	
	def __init__ (self, core, model):
		
		self.core = core
		self.model = model
	
	def init (self):
		
		model_kwargs = {
			
			'use_safetensors': True,
			'torch_dtype': self.core.torch_dtype,
			'token': self.core.args['hf_token'],
			'low_cpu_mem_usage': True,
			'compile': True,
			
		}
		
		transformer_class = self.model.transformer_class ()
		
		if self.core.args['gguf'] != '' and transformer_class is not None:
			
			transformer = self.model.transformer_class ().from_single_file (
				self.core.args['gguf'],
				subfolder = 'transformer',
				quantization_config = GGUFQuantizationConfig (compute_dtype = self.core.torch_dtype),
				torch_dtype = self.core.torch_dtype,
				config = self.core.args['model'],
			)
			
			model_kwargs['transformer'] = transformer
		
		elif self.core.args['nunchaku_transformer'] != '':
			
			transformer = self.model.nunchaku_transformer_class ().from_pretrained (self.core.args['nunchaku_transformer'])
			
			from nunchaku.utils import get_gpu_memory
			
			if get_gpu_memory () <= 18:
				
				# use per-layer offloading for low VRAM. This only requires 3-4GB of VRAM.
				transformer.set_offload (
					True,
					use_pin_memory = False,
					num_blocks_on_gpu = 1,
				)  # increase num_blocks_on_gpu if you have more VRAM
			
			model_kwargs['transformer'] = transformer
		
		else:
			
			if transformer_class is not None:
				
				transformer = transformer_class.from_pretrained (
					self.core.args['model'],
					subfolder = 'transformer',
					torch_dtype = self.core.torch_dtype,
				)
				
				model_kwargs['transformer'] = transformer
		
		if self.core.args['vae'] != '':
			
			model_kwargs['vae'] = AutoencoderKL.from_single_file (
				self.core.args['vae'],
				torch_dtype = self.core.torch_dtype,
			)
		
		print ('Loading model...')
		
		pipe = DiffusionPipeline.from_pretrained (self.core.args['model'], **model_kwargs)
		
		print ('Loading generator...')
		
		pipe.scheduler = self.get_scheduler (pipe.scheduler.config)
		
		if self.core.args['nunchaku_transformer'] != '':
			
			from nunchaku.utils import get_gpu_memory
			
			if get_gpu_memory () <= 18:
				
				pipe.enable_model_cpu_offload ()
				pipe._exclude_from_cpu_offload.append ('transformer')
			
			else:
				pipe.enable_sequential_cpu_offload ()
		
		# else:
		#		pipe.enable_sequential_cpu_offload ()
		
		if self.core.args['lightning_lora'] != '':
			
			model_id, weights_path = self.core.args['lightning_lora'].split (':')
			
			pipe.load_lora_weights (
				model_id,
				weights_path = weights_path,
			)
		
		for lora in self.core.args['lora']:
			
			lora = lora.split (':')
			
			pipe.load_lora_weights (
				lora[0],
				weights_path = lora[1],
				token = lora[3] if len (lora) == 4 else ''
			)
			
			pipe.fuse_lora (lora_scale = lora[2])
			pipe.unload_lora_weights ()
		
		#pipe.vae = AutoencoderTiny.from_pretrained ("D:\\Models\\taesdxl-openvino", torch_dtype = self.core.torch_dtype)
		
		pipe = pipe.to (self.core.device)
		
		return pipe
	
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
		
		config['use_karras_sigmas'] = True if self.core.args['scheduler'] in use_sigmas else False
		
		if self.core.args['scheduler'] in schedulers_cls:
			
			if self.core.args['scheduler'] == 'FlowMatch':
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
			
			scheduler = schedulers_cls[self.core.args['scheduler']].from_config (config)
		
		# elif self.core.args['scheduler'] == 'UniPC':
		#	scheduler = UniPCMultistepScheduler.from_pretrained (self.core.args['model'])
		else:
			
			schedulers_algs = {
				
				'DPM++ 2M SDE': 'sde-dpmsolver++',
				'DPM++ 2M SDE Karras': 'sde-dpmsolver++',
				'DPM++ SDE Karras': 'sde-dpmsolver++',
				
			}
			
			config.euler_at_final = True
			
			scheduler = DPMSolverMultistepScheduler.from_config (config)
			
			if self.core.args['scheduler'] in schedulers_algs:
				scheduler.config.algorithm_type = schedulers_algs[self.core.args['scheduler']]
			else:
				scheduler.config.algorithm_type = self.core.args['scheduler']
		
		return scheduler