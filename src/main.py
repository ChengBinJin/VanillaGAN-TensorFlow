# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf

from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size for one feed forwrad, default: 64')
tf.flags.DEFINE_string('dataset', 'mnist', 'dataset name for choice [mnist|cifar10], default: mnist')
tf.flags.DEFINE_bool('is_train', False, 'training or inference mode, default: False')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100')
tf.flags.DEFINE_integer('iters', 100000, 'number of iterations, default: 100000')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('save_freq', 1000, 'save frequency for model, default: 1000')
tf.flags.DEFINE_integer('sample_freq', 500, 'sample frequency for saving image, default: 500')
tf.flags.DEFINE_integer('sample_size', 16, 'sampling size for check generated image quality, default: 16')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to test, (e.g. 20180704-1736), default: None')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    else:
        solver.test()


if __name__ == '__main__':
    tf.app.run()
