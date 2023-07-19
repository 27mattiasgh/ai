# TODO: Imports
# Keras to create the neural network
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np

# Matplotlib to plot info to show our results

import matplotlib.pyplot as plt


# TODO: Load the MNIST Data
def show_min_max(array, i):
  random_image = array[i]
  print("min and max value in image: ", random_image.min(), random_image.max())

def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(" Digit " + str(labels[i]))
  plt.xticks([])
  plt.yticks([])
  plt.show(block=False)

img_rows, img_cols = 28, 28
num_classes = 10

image_pos = 3534

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_images_backup, train_labels_backup), (test_images_backup, test_labels_backup) = mnist.load_data()
print(train_images.shape)
print(test_images.shape)



train_images = train_images.reshape(train_images.shape[0],  img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0],  img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


plot_image(train_images, image_pos, train_labels)
show_min_max(train_images, image_pos)


train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255


plot_image(train_images, image_pos, train_labels)
show_min_max(train_images, image_pos)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

epochs = 10

model = Sequential()
model.add(Flatten(input_shape=input_shape))

model.add(Dense(1024, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(1024, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))


model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=epochs, shuffle=True)



test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


model.save('mnist_model.h5')
