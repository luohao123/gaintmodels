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

After that, move `stable-diffusion-v1-4` to `weights` folder. Run:

```
python export_df_onnx.py
```

To generate onnx models.