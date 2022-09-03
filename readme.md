# GaintModels

Experiements on testing GaintModels such as GPT3, StableFusion. We offer TensorRT && Int8 quantization on those gaint models. Make you can inference on a 6GB below GPU mem card!


## Install

Some requirements to install:

```
pip install diffusers
pip install transformers
pip install alfred-py
```


## Models


1. `StableFusion`:

First, we need download stablefusion weights from hugging face. 

```
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
git lfs install
cd stable-diffusion-v1-4
git lfs pull
```

You should downloading weights using `git lfs` large file system, the model about `3GB`.

To make `unet_2d_condition` in stablefusion able to export to onnx, make some modification on `diffusers`, following: [link](https://github.com/harishanand95/diffusers/commit/8dd4e822f87e1b4259755a2181218797ceecc410)

file: `diffuers/models/unet_2d_conditions.py`

```
# L137
timesteps = timesteps.broadcast_to(sample.shape[0])
#timesteps = timesteps.broadcast_to(sample.shape[0])
timesteps = timesteps * torch.ones(sample.shape[0])

output = {"sample": sample}
#output = {"sample": sample}

return output
return sample
```

After that, move `stable-diffusion-v1-4` to `weights` folder. Run:

```
python export_df_onnx.py
```

To generate onnx models.