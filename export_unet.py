import onnx
import torch
from diffusers import UNet2DConditionModel
from onnxsim import simplify

'''
unet is the most big part in stable fusion
so we make it runing under tensorrt, it might 
have a best speed on GPU
'''

unet = UNet2DConditionModel.from_pretrained(
    "weights/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16",
    subfolder="unet",
    # use_auth_token=YOUR_TOKEN,
)
unet.cuda()

with torch.inference_mode(), torch.autocast("cuda"):
    inputs = (
        torch.randn(1, 4, 64, 64, dtype=torch.half, device="cuda"),
        torch.randn(1, dtype=torch.half, device="cuda"),
        torch.randn(1, 77, 768, dtype=torch.half, device="cuda"),
    )

    save_f = 'unet_v1_4_fp16_pytorch.onnx'
    save_sim_f = 'sim_unet_v1_4_fp16_pytorch.onnx'

    # Export the model
    torch.onnx.export(
        unet,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        save_f,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input_0", "input_1", "input_2"],
        output_names=["output_0"],
    )

    sim_model, check = simplify(save_f)
    onnx.save(sim_model, save_sim_f)
    print('model saved')
