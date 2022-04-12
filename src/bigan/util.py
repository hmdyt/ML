import matplotlib.pyplot as plt
import torch
import numpy as np

def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    if type(t) == torch.Tensor:
        return t.detach().cpu().numpy()
    elif type(t) != np.ndarray:
        return np.array(t)
    else:
        return t

def judge_is_gray(images: np.ndarray) -> bool:
    if images.shape[3] == 1:
        return True
    elif images.shape[3] == 3:
        return False
    else:
        raise ValueError("images must have 1 or 3 channels")

def show_images64(images: any) -> plt.figure:
    # input shape: (batch_size, n_color, img_size, img_size)
    # matplotlib requirement: (img_size, img_size, n_color)
    images = tensor_to_numpy(images)
    images = np.transpose(images, (0, 2, 3, 1))
    is_gray = judge_is_gray(images)
    fig = plt.figure(figsize=(8,8))
    for i in range(64):
        ax = fig.add_subplot(8, 8, i+1)
        ax.axis('off')
        ax.imshow(images[i], cmap='gray' if is_gray else 'viridis')
    return fig

def show_images_8by2(fixed_img, generated_img) -> plt.figure:
    # input shape: (batch_size, n_color, img_size, img_size)
    # matplotlib requirement: (img_size, img_size, n_color)
    fixed_img = tensor_to_numpy(fixed_img)
    generated_img = tensor_to_numpy(generated_img)
    fixed_img = np.transpose(fixed_img, (0, 2, 3, 1))
    generated_img = np.transpose(generated_img, (0, 2, 3, 1))
    if judge_is_gray(fixed_img) ^ judge_is_gray(generated_img):
        raise ValueError("images must have same channel")
    is_gray = judge_is_gray(fixed_img)
    fig = plt.figure(figsize=(8,2))
    for i in range(8):
        ax = fig.add_subplot(2, 8, i+1)
        ax.axis('off')
        ax.imshow(fixed_img[i], cmap='gray' if is_gray else None)
    for i in range(8, 16):
        ax = fig.add_subplot(2, 8, i+1)
        ax.axis('off')
        ax.imshow(generated_img[i-8], cmap='gray' if is_gray else None)
    return fig

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

def discriminator_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.5)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)