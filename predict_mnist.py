from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image,ImageChops 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(train_images_backup, train_labels_backup), (test_images_backup, test_labels_backup) = mnist.load_data() 

print("CIFAR10 data loaded")

label_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
train_labels_backup = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
test_labels_backup = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 


model = tf.keras.models.load_model('mnist_model.h5')



def predict_image(model, x):
  x = x.astype('float32')
  x = x / 255.0

  x = np.expand_dims(x, axis=0)

  image_predict = model.predict(x, verbose=0)
  print(image_predict, image_predict.shape)
  print("Predicted Label: ", np.argmax(image_predict))

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

path = "six.png"
img = load_img(path, target_size=(28, 28), color_mode="grayscale")
x = img_to_array(img)
true_label = 9



print(x.shape)
p_arr = predict_image(model, x)
plot_value_array(p_arr, true_label, 1)

