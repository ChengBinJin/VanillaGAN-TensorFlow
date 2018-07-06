# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)


class MnistDataset(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.image_size = (28, 28, 1)
        self.num_trains = 0

        self.mnist_path = os.path.join('../../Data', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        print('Load {} dataset...'.format(self.dataset_name))
        self.train_data = input_data.read_data_sets(self.mnist_path, one_hot=True)
        self.num_trains = self.train_data.train.num_examples

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_imgs, _ = self.train_data.train.next_batch(batch_size)

        # reshape 784 vector to 28 x 28 x 1
        batch_imgs = np.reshape(batch_imgs, [batch_size, *self.image_size])
        # imgs = [np.reshape(batch_imgs[idx], self.image_size) for idx in range(batch_imgs.shape[0])]
        # resize to 32 x 32
        # resize_imgs = [cv2.resize(imgs[idx], (32, 32)) for idx in range(len(imgs))]
        # reshape 32 x 32 to 32 x 32 x 1
        # reshape_imgs = [np.reshape(batch_imgs[idx], self.image_size) for idx in range(len(resize_imgs))]
        # list to array
        # arr_imgs = np.asarray(reshape_imgs).astype(np.float32)  # list to array
        return batch_imgs


class Cifar10(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.image_size = (32, 32, 3)
        self.num_trains = 0

        self.cifar10_path = os.path.join('../../Data', self.dataset_name)
        self._load_cifar10()

    def _load_cifar10(self):
        import cifar10

        cifar10.data_path = self.cifar10_path
        print('Load {} dataset...'.format(self.dataset_name))

        # The CIFAR-10 data-set is about 163 MB and will be downloaded automatically if it is not
        # located in the given path.
        cifar10.maybe_download_and_extract()

        self.train_data, _, _ = cifar10.load_training_data()
        self.num_trains = self.train_data.shape[0]
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch(self, batch_size):
        batch_imgs = self.train_data[np.random.choice(self.num_trains, batch_size, replace=False)]
        return batch_imgs


# noinspection PyPep8Naming
def Dataset(dataset_name):
    if dataset_name == 'mnist':
        return MnistDataset(dataset_name)
    elif dataset_name == 'cifar10':
        return Cifar10(dataset_name)
    else:
        raise NotImplementedError

    # tf.logging.set_verbosity(old_v)
