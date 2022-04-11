from ctypes.wintypes import tagPOINT
from typing import Tuple
from torch import Tensor

import numpy as np
import torch
import torchvision

class Generator(torch.nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int], feature_map_len: int) -> None:
        super(Generator, self).__init__()
        self._latent_dim = latent_dim
        self._img_shape = img_shape
        self._feature_map_len = feature_map_len
        def unit_layer(input_channel: int, output_channel: int):
            return [
                torch.nn.ConvTranspose2d(input_channel, output_channel, 4, 1, 0, bias=False),
                torch.nn.BatchNorm2d(output_channel),
                torch.nn.ReLU(True)
            ]
        self.model = torch.nn.Sequential(
            *unit_layer(latent_dim, self._feature_map_len*8),
            *unit_layer(self._feature_map_len*8, self._feature_map_len*4),
            *unit_layer(self._feature_map_len*4, self._feature_map_len*2),
            *unit_layer(self._feature_map_len*2, self._feature_map_len*1),
            torch.nn.ConvTranspose2d(feature_map_len*1, 1, 6, 2, 1, bias=False),
            torch.nn.Tanh()
        )
    def forward(self, z: Tensor) -> Tensor:
        z = z.reshape(z.size(0), self._latent_dim, 1, 1)
        return self.model(z), z

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int], feature_map_len: int) -> None:
        super(Encoder, self).__init__()
        self._latent_dim = latent_dim
        self._img_shape = img_shape
        self._feature_map_len = feature_map_len
        def unit_layer(input_channel: int, output_channel: int):
            return [
                torch.nn.Conv2d(input_channel, output_channel, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(output_channel),
                torch.nn.ReLU(True)
            ]
        self.model = torch.nn.Sequential(
            *unit_layer(1, self._feature_map_len*1),
            *unit_layer(self._feature_map_len*1, self._feature_map_len*2),
            *unit_layer(self._feature_map_len*2, self._feature_map_len*4),
            *unit_layer(self._feature_map_len*4, self._feature_map_len*8),
            torch.nn.Conv2d(self._feature_map_len*8, self._latent_dim, 1, 1, 0, bias=False),
            torch.nn.Tanh()
        )
    def forward(self, x: Tensor) -> Tensor:
        return x, self.model(x.reshape(x.size(0), 1, *self._img_shape)).squeeze()

class Discriminator(torch.nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int], feature_map_len: int) -> None:
        super(Discriminator, self).__init__()
        self._latent_dim = latent_dim
        self._img_shape = img_shape
        self._feature_map_len = feature_map_len
        def unit_layer(input_channel: int, output_channel: int):
            return [
                torch.nn.Conv2d(input_channel, output_channel, 4, 2, 1, bias=False),
                torch.nn.BatchNorm2d(output_channel),
                torch.nn.LeakyReLU(0.2, inplace=True)
            ]
        self.model = torch.nn.Sequential(
            *unit_layer(1, self._feature_map_len*1),
            *unit_layer(self._feature_map_len*1, self._feature_map_len*2),
            *unit_layer(self._feature_map_len*2, self._feature_map_len*4),
            *unit_layer(self._feature_map_len*4, self._feature_map_len*8),
            torch.nn.Conv2d(self._feature_map_len*8, 1, 1, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

# test
if __name__ == '__main__':
    batch_size = 128
    latent_dim = 100
    image_shape = (28, 28)
    feature_map_len = 64
    gen = Generator(latent_dim, image_shape, feature_map_len)
    enc = Encoder(latent_dim, image_shape, feature_map_len)
    z = torch.randn(batch_size, latent_dim, 1, 1)
    genz = gen(z)
    encoded_genz = enc(genz)
    print(genz.shape)
    print(encoded_genz.shape)
    from util import show_images64
    show_images64(genz.reshape(batch_size, *image_shape)).savefig('genz.png')