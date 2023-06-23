import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

class training_chart(object):
    def __init__(self):
        with open('history.pickle', 'rb') as file:
            self.chart = pickle.load(file)

    def plot_model_fit(self):
        if 'accuracy' in self.chart and 'val_accuracy' in self.chart:
            accuracy = self.chart['accuracy']
            val_accuracy = self.chart['val_accuracy']
            plt.plot(accuracy)
            plt.plot(val_accuracy)
            plt.title('Model Fit')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        else:
            print("History data not found. Please train the model first.")
    def loss_view(self):
        if 'loss' in self.chart and 'val_loss' in self.chart:
            loss = self.chart['loss']
            val_loss = self.chart['val_loss']
            plt.plot(loss)
            plt.plot(val_loss)
            plt.title('Model Fit')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        else:
            print("History data not found. Please train the model first.")

proyeksi_gambar_latihan = training_chart()
proyeksi_gambar_latihan.plot_model_fit()
proyeksi_gambar_latihan.loss_view()