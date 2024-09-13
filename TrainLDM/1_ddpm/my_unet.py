from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.attention_processor import Attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d import UNet2DOutput


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_channels: int,
                 num_groups: int) -> None:
        super().__init__()

        self.nonlinearity = nn.SiLU()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.linear = nn.Linear(temb_channels, out_channels)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, t):
        x_input = x
        x_input = self.skip_conv(x_input)

        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)

        t = self.linear(t)
        x = x + t[:, :, None, None]

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        x = x + x_input

        return x


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_channels: int,
                 layers_per_block: int,
                 use_attn: bool,
                 use_downsample: bool,
                 num_groups: int = 32) -> None:
        super().__init__()

        self.res_blocks = []
        self.attns = []

        for i in range(layers_per_block):
            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = out_channels
            self.res_blocks.append(
                ResBlock(_in_channels, out_channels, temb_channels, num_groups))
            if use_attn:
                self.attns.append(
                    Attention(out_channels,
                              heads=8,
                              dim_head=out_channels // 8,
                              residual_connection=True,
                              bias=True,
                              upcast_softmax=True,
                              _from_deprecated_attn_block=True,))
            else:
                self.attns.append(None)

        self.res_blocks = nn.ModuleList(self.res_blocks)
        self.attns = nn.ModuleList(self.attns)
        if use_downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        else:
            self.downsample = None

    def forward(self, x, t):
        res_out = []
        for res_block, attn in zip(self.res_blocks, self.attns):
            x = res_block(x, t)
            if attn is not None:
                x = attn(x)
            res_out.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
            res_out.append(x)
        return x, res_out


class UpsampleLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 skip_channels: int,
                 temb_channels: int,
                 layers_per_block: int,
                 use_attn: bool,
                 use_upsample: bool,
                 num_groups: int = 32) -> None:
        super().__init__()

        self.res_blocks = []
        self.attns = []

        for i in range(layers_per_block):
            if i == 0:
                _in_channels = in_channels
            else:
                _in_channels = out_channels
            if i == layers_per_block - 1:
                _skip_channels = skip_channels
            else:
                _skip_channels = out_channels
            self.res_blocks.append(
                ResBlock(_in_channels + _skip_channels,
                         out_channels, temb_channels, num_groups))
            if use_attn:
                self.attns.append(
                    Attention(out_channels,
                              heads=8,
                              dim_head=out_channels // 8,
                              residual_connection=True,
                              bias=True,
                              upcast_softmax=True,
                              _from_deprecated_attn_block=True,))
            else:
                self.attns.append(None)

        self.res_blocks = nn.ModuleList(self.res_blocks)
        self.attns = nn.ModuleList(self.attns)
        self.upsample = UpsampleLayer(out_channels) if use_upsample else None

    def forward(self, x, res_x, t):
        res_x = reversed(res_x)
        for res_block, attn, res_in in zip(self.res_blocks, self.attns, res_x):
            x = torch.cat((x, res_in), dim=1)
            x = res_block(x, t)
            if attn is not None:
                x = attn(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 temb_channels: int,
                 use_attn: bool,
                 num_groups: int = 32) -> None:
        super().__init__()

        self.res_blocks = []
        self.attns = []

        self.res_blocks.append(
            ResBlock(in_channels, out_channels, temb_channels, num_groups))
        for _ in range(1):
            self.res_blocks.append(
                ResBlock(out_channels, out_channels, temb_channels, num_groups))
            if use_attn:
                self.attns.append(
                    Attention(out_channels,
                              heads=8,
                              dim_head=out_channels // 8,
                              residual_connection=True,
                              bias=True,
                              upcast_softmax=True,
                              _from_deprecated_attn_block=True,))
            else:
                self.attns.append(None)
        self.res_blocks = nn.ModuleList(self.res_blocks)
        self.attns = nn.ModuleList(self.attns)

    def forward(self, x, t):
        x = self.res_blocks[0](x, t)
        for res_block, attn in zip(self.res_blocks[1:], self.attns):
            if attn is not None:
                x = attn(x)
            x = res_block(x, t)

        return x


class MyUnet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_channels: Sequence[int] = [64, 128, 256, 256],
                 has_attn: Sequence[bool] = [False, False, True, False],
                 mid_attn: bool = True,
                 num_groups: int = 32,
                 layers_per_block: int = 2
                 ):
        super().__init__()

        timestep_input_dim = block_channels[0]
        time_embed_dim = block_channels[0] * 4
        self.t_positional = Timesteps(block_channels[0], True, 0)
        self.t_mlp = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        n_blocks = len(block_channels)
        assert n_blocks == len(has_attn)

        self.conv_in = nn.Conv2d(in_channels, block_channels[0], 3, 1, 1)
        self.conv_out_act = nn.SiLU()
        self.conv_out_norm = nn.GroupNorm(num_groups, block_channels[0])
        self.conv_out = nn.Conv2d(block_channels[0], out_channels, 3, 1, 1)

        down_blocks = []
        up_blocks = []
        for i in range(n_blocks):
            if i == 0:
                in_channels = block_channels[0]
            else:
                in_channels = block_channels[i - 1]
            out_channels = block_channels[i]
            use_attn = has_attn[i]
            use_downsample = i != n_blocks - 1
            block = DownBlock(in_channels,
                              out_channels,
                              time_embed_dim,
                              layers_per_block,
                              use_attn,
                              use_downsample,
                              num_groups)
            down_blocks.append(block)
        for i in reversed(range(n_blocks)):
            if i == n_blocks - 1:
                in_channels = block_channels[i]
            else:
                in_channels = block_channels[i + 1]
            if i == 0:
                skip_channels = block_channels[0]
            else:
                skip_channels = block_channels[i - 1]
            out_channels = block_channels[i]
            use_attn = has_attn[i]
            use_upsample = i != 0
            block = UpBlock(in_channels,
                            out_channels,
                            skip_channels,
                            time_embed_dim,
                            layers_per_block + 1,
                            use_attn,
                            use_upsample,
                            num_groups)
            up_blocks.append(block)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.mid_block = MidBlock(block_channels[-1],
                                  block_channels[-1],
                                  time_embed_dim,
                                  mid_attn,
                                  num_groups)

    def forward(self,
                x: torch.Tensor,
                t: Union[torch.Tensor, float, int]) -> UNet2DOutput:
        if not torch.is_tensor(t):
            t = torch.tensor(
                [t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x.device)

        t = t * torch.ones(x.shape[0], dtype=t.dtype,
                           device=t.device)

        t = self.t_positional(t)
        t = self.t_mlp(t)

        x = self.conv_in(x)

        res_out_stack = []
        for down_block in self.down_blocks:
            x, res_out = down_block(x, t)
            res_out_stack += res_out

        x = self.mid_block(x, t)

        for up_block in self.up_blocks:
            num_layers = len(up_block.res_blocks)
            res_x = res_out_stack[-num_layers:]
            res_out_stack = res_out_stack[:-num_layers]
            x = up_block(x, res_x, t)

        x = self.conv_out_norm(x)
        x = self.conv_out_act(x)
        x = self.conv_out(x)
        return UNet2DOutput(sample=x)
