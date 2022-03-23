import torch
import torchvision

from constants import *

# DCGAN論文準拠 Discri, Genのパラメータを初期化する関数


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


class Generator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # 1st ConvTranspose
            torch.nn.ConvTranspose2d(N_Z, N_GENERATOR_FEATURE_MAP*8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(N_GENERATOR_FEATURE_MAP*8),
            torch.nn.ReLU(True),
            # 2nd ConvTranspose
            torch.nn.ConvTranspose2d(N_GENERATOR_FEATURE_MAP*8, N_GENERATOR_FEATURE_MAP*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(N_GENERATOR_FEATURE_MAP*4),
            torch.nn.ReLU(True),
            # 3rd ConvTranspose
            torch.nn.ConvTranspose2d(N_GENERATOR_FEATURE_MAP*4, N_GENERATOR_FEATURE_MAP*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(N_GENERATOR_FEATURE_MAP*2),
            torch.nn.ReLU(True),
            # 4th ConvTranspose
            torch.nn.ConvTranspose2d(N_GENERATOR_FEATURE_MAP*2, N_GENERATOR_FEATURE_MAP*1, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(N_GENERATOR_FEATURE_MAP*1),
            torch.nn.ReLU(True),
            # 5th ConvTranspose
            torch.nn.ConvTranspose2d(N_GENERATOR_FEATURE_MAP*1, N_COLOR, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(torch.nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # 1st conv
            torch.nn.Conv2d(N_COLOR, N_DISCRIMINATOR_FEATURE_MAP, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 2nd conv
            torch.nn.Conv2d(N_DISCRIMINATOR_FEATURE_MAP, N_DISCRIMINATOR_FEATURE_MAP*2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(N_DISCRIMINATOR_FEATURE_MAP*2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 3rd conv
            torch.nn.Conv2d(N_DISCRIMINATOR_FEATURE_MAP*2, N_DISCRIMINATOR_FEATURE_MAP*4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(N_DISCRIMINATOR_FEATURE_MAP*4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 4th conv
            torch.nn.Conv2d(N_DISCRIMINATOR_FEATURE_MAP*4, N_DISCRIMINATOR_FEATURE_MAP*8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(N_DISCRIMINATOR_FEATURE_MAP*8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 5th conv
            torch.nn.Conv2d(N_DISCRIMINATOR_FEATURE_MAP*8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
