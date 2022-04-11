import matplotlib.pyplot as plt
import torch
def show_images64(images: any) -> plt.figure:
    if type(images) == torch.Tensor:
        images = images.squeeze_()
        images = images.detach().cpu().numpy()
    fig = plt.figure(figsize=(8,8))
    for i in range(64):
        ax = fig.add_subplot(8, 8, i+1)
        ax.axis('off')
        ax.imshow(images[i], cmap='gray')
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