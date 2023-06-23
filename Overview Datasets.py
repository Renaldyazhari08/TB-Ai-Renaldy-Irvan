import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Memuat data MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Menampilkan beberapa contoh gambar dari dataset
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title('Label: {}'.format(y_train[i]))
    ax.axis('off')

plt.tight_layout()
plt.show()