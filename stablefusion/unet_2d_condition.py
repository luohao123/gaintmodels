from typing import Dict, Union
from diffusers.models.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionModel,
)
import torch


class UNet2DConditionModelTracable(UNet2DConditionModel):
    def __init__(
        self,
        sample_size=None,
        in_channels=4,
        out_channels=4,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=...,
        up_block_types=...,
        block_out_channels=...,
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=0.00001,
        cross_attention_dim=1280,
        attention_head_dim=8,
    ):
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            center_input_sample,
            flip_sin_to_cos,
            freq_shift,
            down_block_types,
            up_block_types,
            block_out_channels,
            layers_per_block,
            downsample_padding,
            mid_block_scale_factor,
            act_fn,
            norm_num_groups,
            norm_eps,
            cross_attention_dim,
            attention_head_dim,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> Dict[str, torch.FloatTensor]:

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension
        # timesteps = timesteps.broadcast_to(sample.shape[0])
        timesteps = timesteps * torch.ones(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:

            if (
                hasattr(downsample_block, "attentions")
                and downsample_block.attentions is not None
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        # 5. up
        for upsample_block in self.up_blocks:

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if (
                hasattr(upsample_block, "attentions")
                and upsample_block.attentions is not None
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples
                )

        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # output = {"sample": sample}
        # return output
        return sample
