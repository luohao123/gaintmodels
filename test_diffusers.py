import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
device = "cpu"


pipe = StableDiffusionPipeline.from_pretrained('weights/stable-diffusion-v1-4')
pipe = pipe.to(device)

prompt = "naked wonder woman wearing underwear on the beach"
image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
    
image.save("astronaut_rides_horse.png")