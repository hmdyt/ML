import pickle
import matplotlib.pyplot as plt
import torchvision
import numpy as np

imgs = pickle.load(open('pickles/img_list.pickle', 'rb'))

plt.figure(figsize=(6.4, 4.8))
plt.imshow(np.transpose(torchvision.utils.make_grid(imgs, padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()