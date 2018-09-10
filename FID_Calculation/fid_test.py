#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import fid
from scipy.misc import imread
# import imread
# from imread import imread
import tensorflow as tf

gen_path = './gen/' # set path to some generated images

real_path = './real/' # training set statistics



inception_path = './inception/'
inception_path = fid.check_or_download_inception(inception_path) # download inception network

# loads all images into memory (this might require a lot of RAM!)
image_list_gen  = glob.glob(os.path.join(gen_path, '*.jpg'))
image_list_real = glob.glob(os.path.join(real_path, '*.jpg'))
images_gen = np.array([imread(str(fn)).astype(np.float32) for fn in image_list_gen])
images_real = np.array([imread(str(fn)).astype(np.float32) for fn in image_list_real])



fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images_gen, sess, batch_size=100)
    mu_real, sigma_real = fid.calculate_activation_statistics(images_real, sess, batch_size=100)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print("FID: %s" % fid_value)
