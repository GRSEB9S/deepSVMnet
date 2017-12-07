# Bupe Chomba Derreck (BCD)
# December 2017

from layers import Layer
from layers import InputLayer
from layers import DenseRectifiedLinearSVMLayer

import numpy as np

#############################################################################################################

#                                             Model                                                         #
                            
#############################################################################################################

class Model(object):

    @staticmethod
    def predict(model, x):

        size = len(model)

        assert size != 0
            
        model[0].project(x) #pass data to input layer
        
        for i in range(1,size):
            model[i].forward_pass()

        return model[size - 1].output # get output from final layer

    @staticmethod
    def one_hot_to_sign_signal(y):

        l = y.shape[0];
        
        s = np.zeros((l))

        for i in range(l):    
           s[i] = int(2 * y[i] - 1)

        return s 


    # Stochastic Gradient Descent (SGD) + Back Sign Signal Propagation (BSSP) for training Deep Rectified Linear Support Vector Machine(SVM) Networks
    @staticmethod
    def sgd_bssp_train(model, X, Y, epochs):

        size = len(model)

        assert size != 0

        length = size - 1;
        
        for i in range(epochs):

            for x, y in zip(X, Y):
                
               Model.predict(model, x)

               # generate sign signal from one hot encoded y(desired) vector
               back_sign_signal = np.reshape(Model.one_hot_to_sign_signal(y), model[length].output.shape)

               for k in range(0, length):

                   index = length - k
                   
                   layer = model[index] # start from last layer moving towards the input layer
                   
                   grads = layer.evaluate_grads(back_sign_signal) # evaluate param derivatives for the current layer given the back sign signal from the above layer

                   if index > 1: # no need to compute sign signal for fixed input layer
                      back_sign_signal = layer.get_sparse_sign_signal(back_sign_signal) # get sparse sign signal for the layer below

                   layer.update(grads) # update current layer params
