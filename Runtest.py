import numpy as np
import tensorflow as tf 

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Feature scaling and normalization - data collecting
test_images = X_test.reshape((10000, 28, 28, 1)).astype('float32') / 255
test_targets = tf.keras.utils.to_categorical(y_test)
print(f"Shape of test_images: {test_images}")
print(f"Shape of test_targets: {test_targets}")

#Evaluation 
NN = tf.keras.models.load_model("CNN Redigitizer.h5")
loss, accuracy = NN.evaluate(test_images, test_targets)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")