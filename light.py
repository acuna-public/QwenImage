from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionXLPipeline

pipeline = OVStableDiffusionXLPipeline.from_pretrained(
    "D:\\Models\\SDXL-Lightning-2steps-openvino-int8"
)
prompt = "Adorable infant playing with a variety of colorful rattle toys."

images = pipeline(
    prompt=prompt,
    width=512,
    height=512,
    num_inference_steps=2,
    guidance_scale=1.0,
).images
images[0].save("out_image.png")