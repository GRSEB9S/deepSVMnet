# Bupe Chomba Derreck (BCD)
# December 2017

from layers import Layer

from model import Model

import numpy as np

model = []

X = np.array([[0,0],
              [1,0],
              [0,1],
              [1,1]])

Y = np.array([[1],
              [0],
              [0],
              [1]])


np.random.seed(1) # seed random number generator for reproducible results

# Set up model layers
model.append(Layer._input_layer(input_shape = (2,1)))
model.append(Layer.dense(input_layer = model[0], num_nodes = 2, rectified = True))
model.append(Layer.dense(input_layer = model[1], num_nodes = 1, rectified = False)) # We need linear support vector machine output layer | Set rectified to false during training

model[1].learning_rate = 0.02
model[2].learning_rate = 0.02

# train model using the BSSP learning algorithm
Model.sgd_bssp_train(model, X, Y, 5000)

# rectify the output layer | This line can be commented out
model[len(model) - 1].rectified = True

# Test model
for x in X:

  output = Model.predict(model, x)

  print(output)
    
