from qwen_image import Core

Core (
	base_model = 'SDXLTurbo',
	model = 'D:\\Models\\sdxl-turbo-openvino-int8',
	scheduler = 'Euler a',
	steps = 4,
	seed = 42,
	cfg_scale = 1,
	show_prompt = True,
	prompt = '''Adorable infant playing with a variety of {colorful|big} rattle toys.''',
).load ()