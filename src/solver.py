# ---------------------------------------------------------
# Tensorflow Vanilla GAN Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset_ import Dataset
from vanillaGAN import VanillaGAN
import tensorflow_utils as tf_utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset)
        self.model = VanillaGAN(self.sess, self.flags, self.dataset.image_size)

        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.model_out_dir):
                os.makedirs(self.model_out_dir)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                      graph_def=self.sess.graph_def)
        else:
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def train(self):
        for iter_time in range(self.flags.iters):
            # samppling images and save them
            self.sample(iter_time)

            # train_step
            batch_imgs = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
            loss, summary = self.model.train_step(batch_imgs)
            self.model.print_info(loss, iter_time)
            self.train_writer.add_summary(summary, iter_time)
            self.train_writer.flush()

            # save model
            self.save_model(iter_time)

        self.save_model(self.flags.iters)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        num_iters = 10
        total_time = 0.
        for iter_time in range(num_iters):
            # measure inference time
            start_time = time.time()
            imgs = self.model.sample_imgs()  # inference
            total_time += time.time() - start_time
            self.model.plots(imgs, iter_time, self.test_out_dir)

        print('Avg PT: {:.2f} msec.'.format(total_time / num_iters * 1000.))

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            imgs = self.model.sample_imgs()
            self.model.plots(imgs, iter_time, self.sample_out_dir)

    def save_model(self, iter_time):
        if np.mod(iter_time + 1, self.flags.save_freq) == 0:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)

            print('=====================================')
            print('             Model saved!            ')
            print('=====================================\n')

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))
            return True
        else:
            return False
