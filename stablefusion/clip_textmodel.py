from numpy import triu
from transformers import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPTextTransformer
import torch
from torch import nn


def triu_onnx(x, diagonal=0):
    l = x.shape[0]
    arange = torch.arange(l, device=x.device)
    mask = arange.expand(l, l)
    arange = arange.unsqueeze(-1)
    if diagonal:
        arange = arange + diagonal
    mask = mask >= arange
    return mask * x


class Triu(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return triu_onnx(x)


class CIPTextTransformerTracable(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        # mask.triu_(1)  # zero out the lower diagonal
        triu_onnx(mask, 1)
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class CLIPTextModelTracable(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CIPTextTransformerTracable(config)
