#!/usr/bin/env python

from __future__ import absolute_import, print_function, division

import os
import sys
import glob
import time
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('frozen_graph', 'frozen.pb', r"""where the frozen graph is""")

from radio_util import DataProvider
files = glob.glob('../bgs_example_data/seek_cache/*')

# These two functions are copied directly from Unet's implementation
#
# 
def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))

def main(_):
    print(FLAGS.frozen_graph) 

    np.random.seed(42)
    data_provider = DataProvider(10000, files)
    x_test, y_test = data_provider(1)

    with tf.gfile.GFile(FLAGS.frozen_graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        for node in graph_def.node:
            print(node.name)

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map={}, name='')
        predicter = graph.get_tensor_by_name('pixel_wise_softmax_2:0')
        x = graph.get_tensor_by_name('Placeholder:0')
        y = graph.get_tensor_by_name('Placeholder_1:0')
        keep_prob = graph.get_tensor_by_name('Placeholder_2:0')

        with tf.Session() as sess:
            start_time = time.time()
            prediction = sess.run(predicter, feed_dict={x: x_test, y: y_test, keep_prob: 1.})
            duration = time.time() - start_time

            error = error_rate(prediction, crop_to_shape(y_test, prediction.shape))           
            print('Duration: {:.2f} s, Error rate: {:.2f}%'.format(duration, error))


if __name__ == '__main__':
    print('Using TensorFlow version "%s"' % tf.__version__)
    tf.app.run()
