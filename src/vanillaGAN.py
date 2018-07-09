# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------

import collections
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.contrib.layers import flatten

import tensorflow_utils as tf_utils
import utils as utils


class VanillaGAN(object):
    def __init__(self, sess, flags, image_size):
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.num_hiddens = [128, 256]
        self.out_func = tf.nn.tanh if self.flags.dataset == 'cifar10' else tf.nn.sigmoid

        self._build_net()
        self._tensorboard()

        print('Initialized Vanilla GAN SUCCESS!')

    def _build_net(self, is_train=True):
        if is_train:
            self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim])
            self.y_imgs = tf.placeholder(tf.float32, shape=[None, *self.image_size])

            # converting cifar10 dataset to [-1., 1.]
            if self.flags.dataset == 'cifar10':
                self.y_imgs = 2. * self.y_imgs - 1.

            self.g_samples = self.generator(self.z)
            d_real, d_logit_real = self.discriminator(self.y_imgs)
            d_fake, d_logit_fake = self.discriminator(self.g_samples, is_reuse=True)

            # discriminator loss
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logit_real, labels=tf.ones_like(d_logit_real)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake)))
            self.d_loss = d_loss_real + d_loss_fake

            # generator loss
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

            self.dis_optim = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1).\
                minimize(self.d_loss, var_list=d_vars)
            self.gen_optim = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1).\
                minimize(self.g_loss, var_list=g_vars)
        else:
            self.z = tf.placeholder(tf.float32, shape=[None, self.flags.z_dim])
            self.g_samples = self.generator(self.z)

    def _tensorboard(self):
        tf.summary.scalar('loss/d_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()

    def generator(self, x_data, name='g_'):
        with tf.variable_scope(name):
            x_data, output = flatten(x_data), None

            if self.flags.dataset == 'mnist':
                g0 = tf.nn.relu(tf_utils.linear(x_data, self.num_hiddens[0], name='fc1'), name='relu1')
                output = tf_utils.linear(g0, self.image_size[0] * self.image_size[1] * self.image_size[2])
            elif self.flags.dataset == 'cifar10':
                g0 = tf.nn.relu(tf_utils.linear(x_data, self.num_hiddens[0], name='fc1'), name='relu1')
                g1 = tf.nn.relu(tf_utils.linear(g0, self.num_hiddens[1], name='fc2'), name='relu2')
                output = tf_utils.linear(g1, self.image_size[0] * self.image_size[1] * self.image_size[2])
            else:
                raise NotImplementedError

        return self.out_func(output)

    def discriminator(self, y_data, name='d_', is_reuse=False):
        with tf.variable_scope(name, reuse=is_reuse):
            y_data, output = flatten(y_data), None

            if self.flags.dataset == 'mnist':
                d0 = tf.nn.relu(tf_utils.linear(y_data, self.num_hiddens[0], name='fc1'))
                output = tf_utils.linear(d0, 1, name='fc2')
            elif self.flags.dataset == 'cifar10':
                d0 = tf.nn.relu(tf_utils.linear(y_data, self.num_hiddens[0], name='fc1'))
                d1 = tf.nn.relu(tf_utils.linear(d0, self.num_hiddens[1], name='fc2'))
                output = tf_utils.linear(d1, 1, name='fc3')
            else:
                raise NotImplementedError

        return tf.nn.sigmoid(output), output

    def train_step(self, imgs):
        feed = {self.z: self.sample_z(num=self.flags.batch_size),
                self.y_imgs: imgs}

        _, d_loss = self.sess.run([self.dis_optim, self.d_loss], feed_dict=feed)
        _, g_loss = self.sess.run([self.gen_optim, self.g_loss], feed_dict=feed)

        # Run g_optim tweice to make sure that d_loss does not go to zero (different from paper)
        _, g_loss, summary = self.sess.run([self.gen_optim, self.g_loss, self.summary_op], feed_dict=feed)

        return [d_loss, g_loss], summary

    def sample_imgs(self):
        g_feed = {self.z: self.sample_z(num=self.flags.sample_size)}
        y_fakes = self.sess.run(self.g_samples, feed_dict=g_feed)
        # y_fakes = self.sess.run(self.g_samples, feed_dict=g_feed)

        return [y_fakes]

    def sample_z(self, num=64):
        return np.random.uniform(-1.0, 1.0, size=[num, self.flags.z_dim])

    def print_info(self, loss, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('d_loss', loss[0]), ('g_loss', loss[1]),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def plots(self, imgs_, iter_time, save_file):
        # reshape image from vector to (N, H, W, C)
        imgs_fake = np.reshape(imgs_[0], (self.flags.sample_size, *self.image_size))

        imgs = []
        for img in imgs_fake:
            imgs.append(img)

        # parameters for plot size
        scale, margin = 0.05, 0.01
        n_cols, n_rows = int(np.sqrt(len(imgs))), int(np.sqrt(len(imgs)))
        cell_size_h, cell_size_w = imgs[0].shape[0] * scale, imgs[0].shape[1] * scale

        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        imgs = [utils.inverse_transform(imgs[idx]) for idx in range(len(imgs))]

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                if self.flags.dataset == 'cifar10':
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                elif self.flags.dataset == 'mnist':
                    plt.imshow((imgs[row_index * n_cols + col_index]).reshape(
                        self.image_size[0], self.image_size[1]), cmap='Greys_r')
                else:
                    raise NotImplementedError

        plt.savefig(save_file + '/sample_{}.png'.format(str(iter_time)), bbox_inches='tight')
        plt.close(fig)
