from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image,ImageChops 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
(train_images_backup, train_labels_backup), (test_images_backup, test_labels_backup) = cifar10.load_data() 

print("CIFAR10 data loaded")

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
train_labels_backup = [item for sublist in train_labels_backup for item in sublist]
test_labels_backup = [item for sublist in test_labels_backup for item in sublist]


model = tf.keras.models.load_model('model.h5')


def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(" This is " + label_names[labels[i]])
  plt.xticks([])
  plt.yticks([])
  plt.show()

def predict_image(model, x):
  x = x.astype('float32')
  x = x / 255.0

  x = np.expand_dims(x, axis=0)

  image_predict = model.predict(x, verbose=0)
  print("Predicted Label: ", label_names[np.argmax(image_predict)])

  plt.imshow(np.squeeze(x))
  plt.xticks([])
  plt.yticks([])
  plt.show()
 

  return image_predict

def plot_value_array(predictions_array, true_label, h):
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
  plt.ylim([(-1*h), h])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  plt.show()

print('predicting')

path = "horse.png"
img = load_img(path, target_size=(32,32), color_mode = "rgb")
x = img_to_array(img)
true_label = 6


p_arr = predict_image(model, x)
plot_value_array(p_arr, true_label, 1)

