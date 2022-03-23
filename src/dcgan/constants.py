import torch

# constant definitions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# dataloader
IMAGE_SIZE = 64
BATCH_SIZE = 128
N_WORKERS = 4

# train
N_EPOCHS = 5
OPTIM_LR = 0.0002
BETA1 = 0.5
REAL_LABEL = 1.
FAKE_LABEL = 0.
# model
N_Z = 100
N_COLOR = 3
N_GENERATOR_FEATURE_MAP = 64
N_DISCRIMINATOR_FEATURE_MAP = 64
