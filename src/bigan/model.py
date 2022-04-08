from typing import Tuple
from torch import Tensor

import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

from util import show_images64, weights_init, discriminator_weights_init

class Generator(torch.nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int]) -> None:
        super(Generator, self).__init__()
        self._img_shape = img_shape
        def unit_layer(input_len: int, output_len: int, is_normalize: bool = True):
            layers = [torch.nn.Linear(input_len, output_len)]
            if is_normalize:
                layers.append(torch.nn.BatchNorm1d(output_len, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = torch.nn.Sequential(
            *unit_layer(latent_dim, 128),
            *unit_layer(128, 256),
            *unit_layer(256, 512),
            *unit_layer(512, 1024),
            torch.nn.Linear(1024, int(np.prod(img_shape))),
            torch.nn.Tanh()
        )
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        img = self.model(z)
        img = img.view(img.size(0), *self._img_shape)
        return img, z

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int]) -> None:
        super(Encoder, self).__init__()
        self._img_shape = img_shape
        def unit_layer(input_len: int, output_len: int, is_normalize: bool = True):
            layers = [torch.nn.Linear(input_len, output_len)]
            if is_normalize:
                layers.append(torch.nn.BatchNorm1d(output_len, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = torch.nn.Sequential(
            torch.nn.Linear(np.prod(self._img_shape), 1024),
            *unit_layer(1024, 512, is_normalize=True),
            *unit_layer(512, 256, is_normalize=True),
            *unit_layer(256, 128, is_normalize=True),
            *unit_layer(128, latent_dim, is_normalize=True),
            torch.nn.Tanh()
        )
    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.reshape(-1, np.prod(self._img_shape))
        z = self.model(img)
        img = img.view(img.size(0), *self._img_shape)
        return img, z

class Discriminator(torch.nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int]) -> None:
        super(Discriminator, self).__init__()
        joint_len: int = latent_dim + np.prod(img_shape)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(joint_len, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1),
        )
    def forward(self, img: Tensor, z: Tensor) -> Tensor:
        joint = torch.cat((img.view(img.size(0), -1), z), dim=1)
        Y = self.model(joint)
        return Y.squeeze_()

class BiGAN:
    def __init__(
        self,
        n_epochs: int,
        latent_dim: int,
        img_shape: Tuple[int, int],
        lr_EG: float,
        lr_D: float,
        weight_decay: float,
        scheduler_gamma: float,
        fixed_z: Tensor,
        fixed_img: Tensor,
        record_dir: str,
    ):  
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device:", self._device)

        self._n_epochs = n_epochs
        self._latent_dim = latent_dim
        self._img_shape = img_shape
        self._lr_EG = lr_EG
        self._lr_D = lr_D
        self._weight_decay = weight_decay
        self._scheduler_gamma = scheduler_gamma
        self._fixed_z = fixed_z.to(self._device)
        self._fixed_img = fixed_img.to(self._device)
        self._record_dir = record_dir if record_dir[-1] == '/' else record_dir + '/'
        os.makedirs(self._record_dir, exist_ok=True)
        os.makedirs(self._record_dir + 'Generator', exist_ok=True)
        os.makedirs(self._record_dir + 'Encoder', exist_ok=True)


        self._G = Generator(self._latent_dim, self._img_shape).to(self._device)
        self._E = Encoder(self._latent_dim, self._img_shape).to(self._device)
        self._D = Discriminator(self._latent_dim, self._img_shape).to(self._device)
        self._G.apply(weights_init)
        self._E.apply(weights_init)
        self._D.apply(discriminator_weights_init)

        self._criterion = torch.nn.MSELoss()
        self._EG_optim = torch.optim.Adam(
            [{'params': self._G.parameters()}, {'params': self._E.parameters()}],
            lr = self._lr_EG,
            weight_decay = self._weight_decay
        )
        self._D_optim = torch.optim.Adam(
            self._D.parameters(),
            lr = self._lr_D,
            weight_decay = self._weight_decay
        )
        self._EG_scheduler = torch.optim.lr_scheduler.ExponentialLR(self._EG_optim, gamma = self._scheduler_gamma)
        self._D_scheduler = torch.optim.lr_scheduler.ExponentialLR(self._D_optim, gamma = self._scheduler_gamma)

    def _record_loss(self, losses_D, losses_EG):
        axis_array = list(range(len(losses_EG)))
        plt.plot(axis_array, losses_EG, label='loss_EG')
        plt.plot(axis_array, losses_D, label='loss_D')
        plt.legend()
        plt.savefig(self._record_dir + 'loss.png')
        plt.cla()
        plt.clf()
        plt.close()
    
    def _record_G_img(self, epoch):
        self._G.eval()
        img, _ = self._G(self._fixed_z)
        show_images64(img).savefig("{}/Generator/G_{}epoch.png".format(self._record_dir, epoch))
        plt.cla()
        plt.clf()
        plt.close()
    
    def _record_E_img(self, epoch):
        self._G.eval()
        self._E.eval()
        _, generated_z = self._E(self._fixed_img)
        generated_img, _ = self._G(generated_z)
        fig = plt.figure(figsize=(8, 2))
        for i in range(8):
            ax = fig.add_subplot(2, 8, i+1)
            ax.axis('off')
            ax.imshow(self._fixed_img[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        for i in range(8, 16):
            ax = fig.add_subplot(2, 8, i+1)
            ax.axis('off')
            ax.imshow(generated_img[i-8].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        fig.savefig("{}/Encoder/E_{}epoch.png".format(self._record_dir, epoch))
        plt.cla()
        plt.clf()
        plt.close()

    def train(self, train_loader: torch.utils.data.DataLoader) -> None:
        losses_EG = []
        losses_D = []
        self._record_G_img(0)
        self._record_E_img(0)
        for epoch in tqdm(range(self._n_epochs), desc="Epoch", leave=False):
            for i, (imgs, _) in enumerate(tqdm(train_loader, desc='Batches', leave=False)):
                # toggle train mode
                self._G.train()
                self._E.train()
                self._D.train()
                # prepare
                batch_size = imgs.size(0)
                label_real = torch.full((batch_size,), 0.9, dtype=torch.float, device=self._device)
                label_fake = torch.full((batch_size,), 0, dtype=torch.float, device=self._device)
                # train E G
                # forward propagation
                imgs_real, z_real = self._E(imgs.to(self._device))
                predicted_real = self._D(imgs_real, z_real)
                z_fake = torch.randn(batch_size, self._latent_dim, device=self._device)
                imgs_fake, z_fake = self._G(z_fake)
                predicted_fake = self._D(imgs_fake, z_fake)
                # backward propagation
                loss_EG = self._criterion(predicted_real, label_fake) + self._criterion(predicted_fake, label_real)
                loss_EG = loss_EG / 2.
                self._EG_optim.zero_grad()
                loss_EG.backward()
                self._EG_optim.step() 
                # train D
                # forward propagation
                z_fake = torch.randn(batch_size, self._latent_dim, device=self._device)
                imgs_fake, z_fake = self._G(z_fake)
                imgs_real, z_real = self._E(imgs.to(self._device))
                predicted_fake = self._D(imgs_fake, z_fake)
                predicted_real = self._D(imgs_real, z_real)
                # backward propagation
                loss_D = self._criterion(predicted_fake, label_fake) + self._criterion(predicted_real, label_real)
                loss_D = loss_D / 2.
                self._D_optim.zero_grad()
                loss_D.backward()
                self._D_optim.step()
                # record
                losses_EG.append(loss_EG.item())
                losses_D.append(loss_D.item())
                # print
                if i % 100 == 0:
                    print("Epoch: {}, Batch: {}, Loss_EG: {:.4f}, Loss_D: {:.4f}".format(epoch, i, loss_EG.item(), loss_D.item()))
            # scheduler
            self._EG_scheduler.step()
            self._D_scheduler.step()
            # save graph
            self._record_loss(losses_D, losses_EG)
            self._record_G_img(epoch+1)
            self._record_E_img(epoch+1)


            

# test
if __name__ == "__main__":
    BATCH_SIZE = int(2**12)
    train_data = torchvision.datasets.MNIST(
        root = "/home/yuto/repos/ML/data/",
        train = True,
        download = True,
        transform = torchvision.transforms.ToTensor(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 12,
    )
    bigan = BiGAN(
        n_epochs = 100,
        latent_dim = 100,
        img_shape = (28, 28),
        lr_EG = 0.001,
        lr_D = 0.001,
        weight_decay = 0.0,
        scheduler_gamma = 0.99,
        record_dir = "./record_test/",
        fixed_z = torch.randn(64, 100),
        fixed_img = torch.Tensor(next(iter(train_loader))[0])
    )
    bigan.train(train_loader)