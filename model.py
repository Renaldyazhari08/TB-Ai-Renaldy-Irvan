"""
Penyampaian seorang pengajar matematikawan dalam mengajarkan matematika selalu terlibat dengan angka. 
Namun dalam kasus ini setiap pengajar mempunyai tipe tulisan yang berbeda ,
sering kali angka yang ditulis terkadang sulit dibaca.  
Dengan ini tujuan program  Digit recognizer ini dibuat , 
yang memberikan manfaat diantaranya memberikan alat bantu pengajaran yang efektif bagi pengajar matematika, 
dan meningkatkan efisiensi pembelajaran. 
Jadi dengan program ini akan membuat prediksi angka yang digambar dan mendeteksi angka" tersebut.
"""
import numpy as np
import cv2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

class NeuralNet(object):
    def __init__(self):
        # Data collecting
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Data Preparation (feature scaling and normalization)
        self.training_images = X_train.reshape((60000, 28 , 28, 1)).astype('float32') / 255
        self.training_targets = to_categorical(y_train)

        self.test_images = X_test.reshape((10000, 28 , 28, 1)).astype('float32') / 255
        self.test_targets = to_categorical(y_test)

        # Building Model
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))

        # Model Training
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        self.history = self.model.fit(self.training_images, self.training_targets, validation_split=0.2, callbacks=[EarlyStopping(patience=2)], epochs=50)
        self.model.save("CNN Redigitizer.h5")
        self.save_history()

    def save_history(self):
        with open('history.pickle', 'wb') as file:
            pickle.dump(self.history.history, file)

    def load_history(self):
        with open('history.pickle', 'rb') as file:
            self.history = pickle.load(file)

    def plot_model_fit(self):
        if 'accuracy' in self.history.history and 'val_accuracy' in self.history.history:
            accuracy = self.history.history['accuracy']
            val_accuracy = self.history.history['val_accuracy']
            plt.plot(accuracy)
            plt.plot(val_accuracy)
            plt.title('Model Fit')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.show()
        else:
            print("History data not found. Please train the model first.")

    def predict(self, image):
        input = cv2.resize(image, (28, 28)).reshape((1, 28, 28, 1)).astype('float32') / 255
        predictions = self.model.predict(input)
        predicted_class = np.argmax(predictions, axis=1)
        return predicted_class

net = NeuralNet()
net.plot_model_fit()

# Simpan history ke file
net.save_history()

# Menggunakan history yang telah disimpan
net.load_history()
net.plot_model_fit()