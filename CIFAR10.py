# TODO: Imports
# Keras to create the neural network
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np

# Matplotlib to plot info to show our results
import matplotlib.pyplot as plt


# TODO: Load the CIFAR Data
def show_min_max(array, i):
  random_image = array[i]
  print("min and max value in image: ", random_image.min(), random_image.max())

def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(str(label_names[labels[i]]))
  plt.xticks([])
  plt.yticks([])  
  print('ccc')
  plt.show(block=False)


# Variables
img_rows, img_cols = 32, 32
num_classes = 10

# Input Shape
input_shape = (img_rows, img_cols, 3)

# Load the data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
(train_images_backup, train_labels_backup), (test_images_backup, test_labels_backup) = cifar10.load_data()

label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_labels_backup = [item for sublist in train_labels_backup for item in sublist]
test_labels_backup = [item for sublist in test_labels_backup for item in sublist]

print(train_images.shape)
print(test_images.shape)


plot_image(train_images, 1, train_labels_backup)
show_min_max(train_images, 1)



# Data Cleaning
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')



plot_image(train_images, 1, train_labels_backup)
show_min_max(train_images, 1)

train_images /= 255
test_images /= 255

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

print('running network')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

epochs = 1

batch_size = 64


model = Sequential()

#Layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.2))


model.add(BatchNormalization())

#More Layers

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(rate=0.3))
model.add(BatchNormalization())




#Even More Layers
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Dropout(rate=0.3))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()





model.compile(loss=keras.losses.categorical_crossentropy, optimizer = 'adam')
model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_images, test_labels), shuffle=True)

print('evaluation')

scores = model.evaluate(test_images, test_labels,verbose=0)
print('Test accuracy:', scores)


model.save('Python Programming\Python Intro\model.h5')
print('model saved')

