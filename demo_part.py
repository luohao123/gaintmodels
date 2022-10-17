import os
import torch
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from PIL import Image
import argparse

# from alfred.deploy.tensorrt.wrapper import TensorRTInferencer
from stablefusion.trt_model import TRTModel
from alfred import logger


torch_device = "cuda"

YOUR_TOKEN = None
height = 512
width = 512
UNET_INPUTS_CHANNEL = 4
BASE_MODEL_DIR = "weights/stable-diffusion-v1-4"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--beta-start", type=float, default=0.00085, help="beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012, help="::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear")
    parser.add_argument("--trt", action="store_true", default=False)

    parser.add_argument("--num_inference_steps", type=int, default=60)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--eta", type=float, default=0.0, help="eta")

    parser.add_argument("--prompt", type=str, default="prompts.txt", help="prompt")
    parser.add_argument("--init-image", type=str, default=None, help=" image")
    parser.add_argument("--strength", type=float, default=0.5, help="how [0.0, 1.0]")
    parser.add_argument("--mask", type=str, default=None, help="maskial image")
    return parser.parse_args()


def main():
    args = parse_args()
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    batch_size = 1

    if os.path.isfile(args.prompt):
        txts = open(args.prompt, "r").readlines()
        txts = [i.strip() for i in txts]
    else:
        txts = [args.prompt]

    if args.trt:
        unet_trt_enigne = "unet_fp16.trt"
        logger.info(f"using TensorRT inference unet: {unet_trt_enigne}")
        assert os.path.exists(unet_trt_enigne), f"{unet_trt_enigne} not found!"
        # unet = TensorRTInferencer(unet_trt_enigne)
        unet = TRTModel(unet_trt_enigne)
        logger.info("unet loaded in trt.")
    else:
        unet = UNet2DConditionModel.from_pretrained(
            BASE_MODEL_DIR,
            subfolder="unet",
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token=YOUR_TOKEN,
        )

    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL_DIR, subfolder="vae", use_auth_token=YOUR_TOKEN
    )
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_DIR + "/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_DIR + "/text_encoder")

    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    # Set the models to your inference device
    vae.to(torch_device)
    text_encoder.to(torch_device)
    if not args.trt:
        unet.to(torch_device)

    for index, prompt in enumerate(txts):
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).half().cuda()

        latents = torch.randn(
            (batch_size, UNET_INPUTS_CHANNEL, height // 8, width // 8)
        )
        latents = latents.to(torch_device)
        scheduler.set_timesteps(num_inference_steps)
        latents = latents * scheduler.sigmas[0]

        scheduler.set_timesteps(num_inference_steps)
        # Denoising Loop
        with torch.inference_mode(), autocast("cuda"):
            for i, t in tqdm(enumerate(scheduler.timesteps)):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                sigma = scheduler.sigmas[i]
                latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                if args.trt:
                    # noise_pred = unet.infer(
                    #     latent_model_input, t, encoder_hidden_states=text_embeddings
                    # )
                    inputs = [
                        latent_model_input,
                        torch.tensor([t]).to(torch_device),
                        text_embeddings,
                    ]
                    noise_pred, duration = unet(inputs, timing=True)
                    noise_pred = torch.reshape(
                        noise_pred[0], (batch_size * 2, 4, 64, 64)
                    )
                else:
                    noise_pred = unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents)

        image = image.sample
        # Convert the image with PIL and save it
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save(f"image_generated_{index}.png")


if __name__ == "__main__":
    main()
