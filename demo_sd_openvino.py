import argparse
import os
from stablefusion.stablefusion_ov_engine import StableDiffusionEngine
from diffusers import LMSDiscreteScheduler, PNDMScheduler
import cv2
import numpy as np
from alfred import logger
import os


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    if args.init_image is None:
        scheduler = LMSDiscreteScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            tensor_format="np",
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            skip_prk_steps=True,
            tensor_format="np",
        )
    engine = StableDiffusionEngine(
        model=args.model,
        scheduler=scheduler,
        tokenizer=args.tokenizer,
        local_model_path="weights/onnx",
    )
    txts = []
    if os.path.isfile(args.prompt):
        txts = open(args.prompt, "r").readlines()
        txts = [i.strip() for i in txts]
    else:
        txts = [args.prompt]

    for i, prompt in enumerate(txts):
        image = engine(
            prompt=prompt,
            init_image=None if args.init_image is None else cv2.imread(args.init_image),
            mask=None if args.mask is None else cv2.imread(args.mask, 0),
            strength=args.strength,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
        )
        cv2.imwrite(f"res{i}_{args.output}", image)
        logger.info(f"result save into: res{i}_{args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument(
        "--model",
        type=str,
        default="bes-dev/stable-diffusion-v1-4-openvino",
        help="model name",
    )
    # randomizer params
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for generating consistent images per prompt",
    )
    # scheduler params
    parser.add_argument(
        "--beta-start",
        type=float,
        default=0.00085,
        help="LMSDiscreteScheduler::beta_start",
    )
    parser.add_argument(
        "--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end"
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="scaled_linear",
        help="LMSDiscreteScheduler::beta_schedule",
    )
    # diffusion params
    parser.add_argument(
        "--num-inference-steps", type=int, default=32, help="num inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="guidance scale"
    )
    parser.add_argument("--eta", type=float, default=0.0, help="eta")
    # tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="tokenizer",
    )
    # prompt
    parser.add_argument(
        "--prompt", type=str, default="prompts.txt", help="prompt",
    )
    # img2img params
    parser.add_argument(
        "--init-image", type=str, default=None, help="path to initial image"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help="how strong the initial image should be noised [0.0, 1.0]",
    )
    # inpainting
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="mask of the region to inpaint on the initial image",
    )
    # output name
    parser.add_argument(
        "--output", type=str, default="output.png", help="output image name"
    )
    args = parser.parse_args()
    main(args)
