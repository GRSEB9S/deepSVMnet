# Bupe Chomba Derreck (BCD)
# December 2017

import numpy as np

from layers import Layer

from model import Model

from mnist import*

    
#######################################################################

x_train, y_train, x_test, y_test = get_mnist()

image_count = 55000

X = x_train[0:image_count]
Y = y_train[0:image_count]

model = []

np.random.seed(1) # seed random number generator for reproducible results

# Set up model layers
model.append(Layer._input_layer(input_shape = (1,28,28)))

model.append(Layer.conv2d(input_layer = model[0], kernel_shape = (5,5), num_kernels = 16, stride = 1, padding = 2))
model.append(Layer.pool2d(input_layer = model[1], pool_shape = (2,2), stride = 2))

model.append(Layer.conv2d(input_layer = model[2], kernel_shape = (5,5), num_kernels = 32, stride = 1, padding = 2))
model.append(Layer.pool2d(input_layer = model[3], pool_shape = (2,2), stride = 2))

model.append(Layer.dense(input_layer = model[4], num_nodes = 64, rectified = True))
model.append(Layer.dense(input_layer = model[5], num_nodes = 10, rectified = False)) # We need linear support vector machine output layer | Set rectified to false during training

# set learning rates per layer
model[1].learning_rate = 0.02
model[2].learning_rate = 0.02

model[3].learning_rate = 0.02
model[4].learning_rate = 0.02

model[5].learning_rate = 0.02
model[6].learning_rate = 0.02

#traing model
print("Training model:")
Model.sgd_bssp_train(model, X, Y, 10)

# evaluate model performance
print("Evaluating model:")

correct = 0.0

total = 0.0

for x, y in zip(x_test, y_test):

  y_ = Model.predict(model, x)

  if np.argmax(y) == np.argmax(y_):
      correct += 1.0

  total += 1.0

assert total != 0.0

accuracy  = correct / total

print("Model accuracy:" + str(accuracy))
