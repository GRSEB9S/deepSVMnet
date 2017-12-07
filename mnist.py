# Bupe Chomba Derreck (BCD)
# December 2017

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

def get_mnist():
    
   mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

   x_train = np.array([img.reshape(1,28,28) for img in mnist.train.images])
   y_train = mnist.train.labels

   x_test = np.array([img.reshape(1,28,28) for img in mnist.test.images])
   y_test = mnist.test.labels

   return x_train, y_train, x_test, y_test
