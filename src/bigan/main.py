import argparse
import time

import torch

from model import BiGAN
from DataSelector import DataSelector

torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--n_epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=64)
parser.add_argument("-l", "--latent_dim", type=int, default=100)
parser.add_argument("--img_size", type=int, default=28)
parser.add_argument("--n_color", type=int, default=1, help="number of color channels")
parser.add_argument("--lr_EG", type=float, default=0.0002)
parser.add_argument("--lr_D", type=float, default=0.0002)
parser.add_argument("--weight_decay", type=float, default=2.5e-5)
parser.add_argument("--use_scheduler", type=bool, default=True)
parser.add_argument("--scheduler_gamma", type=float, default=0.99)
parser.add_argument("-r", "--record_dir", type=str, default="record_tmp_{}/".format(str(time.time())))
parser.add_argument("-d", "--dataset_dir", type=str, default="data_tmp/")
parser.add_argument("-n", "--dataset_name", type=str, default="mnist")
parser.add_argument("--num_worker", type=int, default=12)
parser.add_argument("--model_type", type=str, default="linear", help='linear or cnn')
parser.add_argument("--feature_map_len", type=int, default=64)
opts = parser.parse_args()

print(opts)

# init dataloader
data_selector = DataSelector(opts)
train_loader = data_selector.get_loader()
fixed_imgs = torch.Tensor(next(iter(train_loader))[0])

# init model
bigan = BiGAN(
    n_epochs = opts.n_epochs,
    latent_dim = opts.latent_dim,
    img_shape = (opts.img_size, opts.img_size),
    n_color = opts.n_color,
    lr_EG = opts.lr_EG,
    lr_D = opts.lr_D,
    weight_decay = opts.weight_decay,
    use_scheduler = opts.use_scheduler,
    scheduler_gamma = opts.scheduler_gamma,
    record_dir = opts.record_dir,
    fixed_z = torch.randn(64, opts.latent_dim),
    fixed_img = fixed_imgs,
    model_type = opts.model_type,
    feature_map_len = opts.feature_map_len,
)

# start training
bigan.train(train_loader)
print("Training finished!")