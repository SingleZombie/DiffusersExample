import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class Discriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels=3, hidden_channels=512, depth=6,
                 use_bn=False):
        super().__init__()

        use_bias = not use_bn
        norm_cls = nn.BatchNorm2d

        nonlinearity = nn.LeakyReLU(0.2, True)

        d = max(depth - 3, 3)
        layers = [
            nn.Conv2d(in_channels, hidden_channels // (2**d),
                      kernel_size=4, stride=2, padding=1),
            nonlinearity,
        ]
        for i in range(depth - 1):
            c_in = hidden_channels // (2 ** max((d - i), 0))
            c_out = hidden_channels // (2 ** max((d - 1 - i), 0))

            layers.append(
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2,
                          padding=1, bias=use_bias))
            layers.append(norm_cls(c_out))
            layers.append(nonlinearity)
        c_in = c_out
        c_out = hidden_channels
        layers.append(
            nn.Conv2d(
                c_in, c_out, kernel_size=4, stride=1, padding=1, bias=use_bias
            )
        )
        layers.append(norm_cls(c_out))
        layers.append(nonlinearity)
        layers.append(nn.Conv2d(
            c_out, 1, kernel_size=4, stride=1, padding=1
        ))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x
