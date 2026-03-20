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

  ### Generate image:

  ```
    python qwen_image.py \
      --prompt "Girl with `Qwen Image` sign" \
      --ratio "16:9"
  ```
  ### Edit image
  
  ```
    python qwen_image.py \
      --image "C:\QwenExamples\person.jpg" \
      --image "C:\QwenExamples\bedroom.jpg" \
      --prompt "Place person on first image to the room on the second image" \
      --ratio "16:9"
  ```
