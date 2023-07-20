from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

(train_images, train_labels), (_, _) = cifar10.load_data()
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = tf.keras.models.load_model('model.h5')


def plot_image(array, i, labels):
    plt.imshow(np.squeeze(array[i]))
    plt.title("This is " + label_names[labels[i]])
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
    plt.ylim([(-1 * h), h])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    plt.show()


def process_images(image_dir):
    image_files = os.listdir(image_dir)

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        img = load_img(image_path, target_size=(32, 32), color_mode="rgb")
        x = img_to_array(img)
        true_label = 6

        print(f"Processing Image {i+1}: {image_file}")
        p_arr = predict_image(model, x)
        plot_value_array(p_arr, true_label, 1)


# Replace "path_to_images_folder" with the path to the folder containing the images you want to predict.
images_folder = "cifar_images"
process_images(images_folder)
