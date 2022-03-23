#!/usr/bin/env python3

import sys

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from models import Generator
from constants import *

n_batch = int(sys.argv[1])

generator_model = Generator(0).to("cpu")
generator_model.load_state_dict(torch.load("gan_scrath_gen.pth", map_location=torch.device('cpu')))

z = torch.randn(n_batch, N_Z, 1, 1, device=DEVICE)
output = generator_model(z)

plt.imshow(np.transpose(torchvision.utils.make_grid(output, normalize=True).cpu(), (1, 2, 0)))
plt.show()
