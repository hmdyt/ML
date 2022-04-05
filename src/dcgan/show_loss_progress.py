import pickle
import matplotlib.pyplot as plt

loss_discri = pickle.load(open('pickles/losses_discri.pickle', 'rb'))
loss_gen = pickle.load(open('pickles/losses_gen.pickle', 'rb'))

plt.plot(range(len(loss_gen)), loss_gen, label="loss_G")
plt.plot(range(len(loss_discri)), loss_discri, label="loss_D")
plt.legend()
plt.show()