from stablefusion.stablefusion_ov_engine import StableDiffusionEngine
from diffusers.pipelines import StableDiffusionPipeline
import torch
import cv2


def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "weights/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]
    image.save("res.png")
    # cv2.imwrite(f"res.png", image.cpu().numpy())


if __name__ == '__main__':
    main()
