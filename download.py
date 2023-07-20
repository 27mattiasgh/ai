# Imports
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import os


def save_random_cifar_images(num_images=10):
    (train_images, _), (_, _) = cifar10.load_data()


    save_dir = 'cifar_images'
    os.makedirs(save_dir, exist_ok=True)


    random_indices = np.random.randint(0, len(train_images), size=num_images)

    for i, idx in enumerate(random_indices):
        image = train_images[idx]
        file_path = os.path.join(save_dir, f'image_{i + 1}.png')
        tf.keras.preprocessing.image.save_img(file_path, image)


save_random_cifar_images(25)
