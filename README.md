# QwenImage

QwenImage is a CLI version for working with **Qwen Image** and **Qwen Image Edit** models, no more ComfyUI with its clymsy interface, broken nodes and unknown dependencies, just a few CLI strings commands for the same functionality and automate tasks.

## Scopes

- **Qwen Image** and **Qwen Image Edit** support for working with several images pipelines
- Lighting LoRA support by default for reduce steps and encrease quality
- Aspect ratio support (`--ratio` argument)
- Seed, steps number, CFG Ratio, etc. support
- Automatic Huggingface models downloading

## Install

  ```
  git clone https://github.com/acuna-public/QwenImage.git
  ```
  ```
  pip install -r requirements.txt
  ```

In protected environment you need to install all dependencies in [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

## Quick Start

  ### Generate image

  ```
    python qwen_image.py \
      --prompt "Girl with `Qwen Image` sign" \
      --ratio "16:9"
  ```
  ### Edit image

  #### Images as files
  
  ```
    python qwen_image.py \
      --image "C:\QwenExamples\person.jpg" \
      --image "C:\QwenExamples\bedroom.jpg" \
      --prompt "Place person on first image to the room on the second image" \
      --ratio "16:9"
  ```

#### Images as folders

  ```
    python qwen_image.py \
      --image "C:\QwenExamples\Persons" \
      --image "C:\QwenExamples\PosesSet1" \
      --image "C:\QwenExamples\PosesSet2" \
      --prompt "Draw person from first image with pose from second image" \
  ```

Where every person image from first folder will be drawn with poses from first poses set, and after that with poses from second poses set etc which can be useful while using different poses sets for every image.

## Reference

```
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Full path to Qwen Image model. If not set - it will be
                        downloaded automatically.
  --width WIDTH         Generated image width (1024 by default). Ignored when
                        --ratio is set.
  --height HEIGHT       Generated image height (1024 by default). Ignored when
                        --ratio is set.
  --ratio RATIO         Aspect ratio as string (optional). Ignored when
                        --width and --height are set.
  --lightning-lora LIGHTNING_LORA
                        Lightning lora (local path or Huggingface model id)
                        for reduce inference steps number and increase details
                        (recommended)
  --image IMAGE         Images folder of image file (can be multiple)
  --prompt PROMPT, -p PROMPT
                        Prompt (if not set - folder with images and prompts in
                        txt files with same names as every image needed in
                        main folder)
  --negative-prompt NEGATIVE_PROMPT, -n NEGATIVE_PROMPT
                        Negative prompt (optional)
  --positive-magic POSITIVE_MAGIC
                        Positive magic tags for more realism
  --seed SEED, -s SEED  Seed (random by default)
  --hf-token HF_TOKEN   Huggingface token. Set it if models downloading is
                        slow or stuck.
  --cfg-scale CFG_SCALE
                        CFG scale (LoRA according by default)
  --steps STEPS         Inference steps number (LoRA according by default)
  --guidance-scale GUIDANCE_SCALE
                        Guidance scale (1.0 by default)
  --images-per-prompt IMAGES_PER_PROMPT
                        Images number per prompt (1 by default)
  --output-name OUTPUT_NAME
                        Result image name if --image is file
  --verbose, -v         Verbose mode
```
