
import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps
import os

# Paths
model_path = "/home/henry/Documents/Projects/Python/MyScreen OfflineImageGenerator/coamixsd15_v10.safetensors"
base_image_path = "/home/henry/Pictures/HenJay/CelAS/Cel_20260612_022646_Seed-3462628548599614331_Den-0.75_CFG-7.0_Stp-20.png"
output_path = "/home/henry/Pictures/HenJay/CelAS/Cel_20260612_Suggestive_HandPull.png"

# Prompts
pos_prompt = "nude, nsfw, 1girl, realistic, photorealistic, raw photo, high quality, highly detailed skin, detailed pores, highly detailed eyes, cute face, young adult, Gen-Z style, wearing a very loose oversized off-shoulder sweater, hand pulling down sweater, grabbing sweater to reveal breasts, pulling clothing down, revealing deep cleavage and nipples, no bra, messy bun, looking at viewer with a devious smirk, mischievous look, suggestive gaze, biting lip, blushing slightly, playful yet naughty expression, provocative pose, soft natural lighting, depth of field, masterpiece, best quality, ultra-detailed, highres, 8k resolution, cinematic lighting, soft lighting, bokeh, professional photography, sharp focus"
neg_prompt = "worst quality, low quality, normal quality, ugly, blurry, mutated, poorly drawn, extra limbs, bad anatomy, missing fingers, jpeg artifacts, watermark, signature, drawing, illustration, 3d, render, cgi, deformed, distorted, plastic skin, cringe, 3D, clothes, fully dressed"

# Settings
steps = 20
cfg = 7.0
denoise = 0.83
width = 512
height = 768

print("Loading AI Brain... 🧠")
pipe = StableDiffusionImg2ImgPipeline.from_single_file(
    model_path, 
    safety_checker=None, 
    requires_safety_checker=False, 
    torch_dtype=torch.float32,
    local_files_only=True
).to("cpu")

# Set Sampler to Euler A
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Prepare Base Image
raw_img = Image.open(base_image_path).convert("RGB")
img = ImageOps.fit(raw_img, (width, height), Image.Resampling.LANCZOS)

print("Cooking the naughty stuff... 🍳🔥")
result = pipe(
    prompt=pos_prompt,
    negative_prompt=neg_prompt,
    image=img,
    guidance_scale=cfg,
    strength=denoise,
    num_inference_steps=steps
).images[0]

result.save(output_path)
print(f"Done! Image saved to {output_path} 🎀✨")
