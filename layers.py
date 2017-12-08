# Bupe Chomba Derreck (BCD)
# December 2017

import numpy as np
    
#############################################################################################################

#                                           Layer                                                           #
                            
#############################################################################################################

class Layer(object):
    
    def __init__(self):
        self.input_layer = None
        self.weights = None
        self.biases = None
        self.output = None
        self.learning_rate = 0.02
        self.regularization_factor = 0.0

    @staticmethod
    def conv2d(input_layer, kernel_shape, num_kernels, stride = 1, padding = 0):
        return ConvolutionalRectifiedLinearSVMLayer(input_layer, kernel_shape, num_kernels, stride, padding)
    
    @staticmethod
    def dense(input_layer, num_nodes, rectified = True):
        return DenseRectifiedLinearSVMLayer(input_layer, num_nodes, rectified)

    @staticmethod
    def pool2d(input_layer, pool_shape = (2,2), stride = 1):
        return PoolingLayer(input_layer, pool_shape, stride)

    @staticmethod
    def _input_layer(input_shape):
        return InputLayer(input_shape)

    def update(self, grad_buffer):
        pass

    def evaluate_grads(self, target_sign_signal):
        pass
    
    def get_sign_signal(self, back_sign_signal):
        pass

    def project(self, x):
        pass

    def forward_pass(self):
        pass

   
#############################################################################################################

#                                         Input Layer                                                       #
                            
#############################################################################################################

class InputLayer(Layer):
    
   def __init__(self, input_shape):
       super().__init__()
       self.output = np.zeros(input_shape)
    
   def project(self, x):

       self.output = x



#############################################################################################################

#                     Dense Rectified Linear Support Vector Machine Layer                                   #
                            
#############################################################################################################

class DenseRectifiedLinearSVMLayer(Layer):

    def __init__(self, input_layer, num_nodes, rectified = True):
        super().__init__()
        self.input_layer = input_layer
        self.weights = np.random.randn(num_nodes, shape_size(input_layer.output.shape)) 
        self.output = np.zeros((num_nodes, 1, 1))
        self.rectified = rectified
        self.biases = np.full((num_nodes), 0.2) # initialize biases with small positive values | To avoid dead neurons

        w = flatten(self.weights)
        np.absolute(w,w) # positive initial weights only | To avoid dead neurons

    def update(self, grad_buffer):
        
        weight_grads = grad_buffer["weight_grads"]
        bias_grads = grad_buffer["bias_grads"]

        self.weights += self.learning_rate*weight_grads
        self.biases += self.learning_rate*bias_grads

    def evaluate_grads(self, back_sign_signal):

        weight_grads = np.zeros((self.weights.shape))
        bias_grads = np.zeros((self.biases.shape))

        channel_size = back_sign_signal.shape[1] * back_sign_signal.shape[2]

        sign_signal_buffer = flatten(back_sign_signal)

        output_buffer = flatten(self.output)

        input_buffer = flatten(self.input_layer.output)
        
        for i, target_sign in enumerate(sign_signal_buffer):

            kernel_id = int(i / channel_size)
            
            grads = weight_grads[kernel_id]

            params = self.weights[kernel_id]

            if target_sign != 0 and output_buffer[i] * target_sign < 1.0:

                grads += target_sign * input_buffer

                grads -= target_sign * self.regularization_factor * params
                  
                bias_grads[kernel_id] += target_sign
                
        return {'weight_grads':weight_grads, 'bias_grads': bias_grads}


    def get_sparse_sign_signal(self, back_sign_signal):

        deltas = np.zeros(self.input_layer.output.shape)

        input_buffer = flatten(self.input_layer.output)

        sign_signal_buffer = flatten(back_sign_signal)

        output_buffer = flatten(self.output)

        deltas_buffer = flatten(deltas)

        for j, sign_signal in enumerate(sign_signal_buffer):

            if sign_signal != 0 and output_buffer[j] * sign_signal < 1.0:

                deltas_buffer += self.weights[j] * sign_signal
                
        for i, x in enumerate(input_buffer):

           if x <= 0.0:
               deltas_buffer[i] = 0.0
    
        return competitive_credit_assign(deltas)
    
    def project(self, data_point):

        x = data_point.reshape(-1,1)

        out = self.output.reshape(self.weights.shape[0], 1)
        
        np.dot(self.weights, x, out)

        out += self.biases.reshape(out.shape)

        if self.rectified:
          rectify(out)
        
    def forward_pass(self):
        self.project(self.input_layer.output)


#############################################################################################################

#                 Convolutional Rectified Linear Support Vector Machine Layer                               #
                            
#############################################################################################################

class ConvolutionalRectifiedLinearSVMLayer(Layer):

    def __init__(self, input_layer, kernel_shape, num_kernels, stride, padding):
        super().__init__()
        self.input_layer = input_layer
        self.kernel_shape = kernel_shape
        self.num_kernels = num_kernels
        self.stride = stride
        self.padding = padding
        self.padded_input = None
       
        depth = input_layer.output.shape[0]
        
        self.weights = np.random.randn(num_kernels, depth, kernel_shape[0], kernel_shape[1]) 
        self.biases = np.full((num_kernels), 0.2) # initialize biases width small positive values
 
        w = flatten(self.weights)
        np.absolute(w,w) # positive initial weights only | To avoid dead neurons

        self.height_out, self.width_out = compute_output_dimension(input_layer.output.shape, kernel_shape, stride, padding)

        self.output = np.zeros((self.num_kernels, self.height_out, self.width_out))

        self.cols_buffer = np.zeros((depth * shape_size(kernel_shape), self.height_out*self.width_out))
        
    def update(self, grad_buffer):
        
        weight_grads = grad_buffer["weight_grads"]
        bias_grads = grad_buffer["bias_grads"]

        self.weights += self.learning_rate*weight_grads
        self.biases += self.learning_rate*bias_grads

    #in this case it is a convolution operation
    def project(self, x):

        self.padded_input = pad(x, self.padding)
        
        im2col(self.padded_input, self.num_kernels, self.kernel_shape, self.stride, self.cols_buffer)

        w = self.weights.reshape(self.num_kernels, -1)

        self.output = self.output.reshape(self.num_kernels, -1)

        np.dot(w, self.cols_buffer, self.output)

        for i in range(self.num_kernels):

             s = self.output[i]
             
             s += self.biases[i]

             rectify(s)
            
        self.output = self.output.reshape(self.num_kernels, self.height_out, self.width_out)


    def forward_pass(self):
        self.project(self.input_layer.output)

    def evaluate_grads(self, back_sign_signal):

        weight_grads = np.zeros((self.weights.shape))
        bias_grads = np.zeros((self.biases.shape))

        channel_size = back_sign_signal.shape[1] * back_sign_signal.shape[2]

        sign_signal_buffer = back_sign_signal.reshape(-1, channel_size)

        output_buffer = self.output.reshape(-1, channel_size)

        norm_factor = np.zeros((self.num_kernels))

        for kernel_id in range(self.num_kernels):

            sign_signal_slice = sign_signal_buffer[kernel_id]
            output_slice = output_buffer[kernel_id]
            
            for n, a in enumerate(sign_signal_slice):
                
               if a != 0 and output_slice[n] * a < 1.0:
                  norm_factor[kernel_id] += 1.0
	       

        for kernel_id in range(self.num_kernels):

            grads = flatten(weight_grads[kernel_id])

            params = flatten(self.weights[kernel_id])

            norm_val = norm_factor[kernel_id]

            sign_signal_slice = sign_signal_buffer[kernel_id]

            output_slice = output_buffer[kernel_id]
                
            for i, target_sign in enumerate(sign_signal_slice):

                if target_sign != 0 and output_slice[i] * target_sign < 1.0:

                    s = target_sign / norm_val

                    grads += s * self.cols_buffer[:,i] 

                    grads -= s * self.regularization_factor * params
                      
                    bias_grads[kernel_id] += s
                 
        return {'weight_grads':weight_grads, 'bias_grads': bias_grads}


    def get_sparse_sign_signal(self, back_sign_signal):

        deltas = np.zeros(self.padded_input.shape)

        input_buffer = flatten(self.padded_input)

        sign_signal_buffer = flatten(back_sign_signal)

        output_buffer = flatten(self.output)

        deltas_buffer = flatten(deltas)

        kernel_height, kernel_width = self.kernel_shape

        height_in = deltas.shape[1]
        width_in = deltas.shape[2]
        
        size = self.height_out * self.width_out

        for j in range(self.num_kernels):

            col = 0
            row = 0
            
            sign_signal_slice = sign_signal_buffer[j * size : (j + 1) * size]
            output_slice = output_buffer[j * size : (j + 1) * size]
            kernel_params = self.weights[j]
            
            for i, sign_signal in enumerate(sign_signal_slice):

               if sign_signal != 0 and output_slice[i] * sign_signal < 1.0:
                   
                  deltas[:,row:kernel_height+row, col:kernel_width + col] += sign_signal * kernel_params
               
               col += self.stride

               if (kernel_width + col) > width_in:
                   col = 0
                   row += self.stride

               if (kernel_height + row) > height_in:
                   break
        
        for i, x in enumerate(input_buffer):

           if x <= 0.0:
               deltas_buffer[i] = 0.0

        return competitive_credit_assign(deltas, self.padding)

#############################################################################################################

#                                      Pooling Layer                                                        #
                            
#############################################################################################################

class PoolingLayer(Layer):

    def __init__(self, input_layer, pool_shape, stride):
        super().__init__()
        self.input_layer = input_layer
        self.pool_shape = pool_shape
        self.stride = stride
        
        depth = input_layer.output.shape[0]
        
        self.height_out, self.width_out = int(input_layer.output.shape[1] / stride), int(input_layer.output.shape[2] / stride)

        self.output = np.zeros((depth, self.height_out, self.width_out))

        self.max_indices = np.full((depth, self.height_out, self.width_out, 2),-1)

    #in this case it is a convolution operation
    def project(self, x):

        depth, height_in, width_in = x.shape

        size = self.height_out * self.width_out

        out = self.output.reshape((depth, size))
 
        indices = self.max_indices.reshape((depth, size, 2))

        # reset the buffers
        out.fill(0.0)
        indices.fill(-1)
        
        pool_height, pool_width = self.pool_shape
        
        for j in range(depth):
            
            row = 0
            col = 0
            
            for i in range(size):

                row_end = pool_height + row
                col_end = pool_width + col

                if row_end >= height_in:
                    row_end = height_in

                if col_end >= width_in:
                    col_end = width_in

                sample = x[j][row:row_end, col:col_end]

                max_val = np.max(sample)

                if max_val > 0:

                   index = np.where(sample == max_val)

                   indices[j][i] = [index[0][0] + row, index[1][0] + col] #keep track of max activating nodes

                   out[j][i] = max_val
                    
                col += self.stride

                if col >= width_in:
                   col = 0
                   row += self.stride

                if row >= height_in:
                    break


    def forward_pass(self):
        self.project(self.input_layer.output)

    
    def get_sparse_sign_signal(self, back_sign_signal):

        depth, height, width = back_sign_signal.shape

        size = width * height
        
        forward_sign_signal = np.zeros(self.input_layer.output.shape)

        out_buffer = self.output.reshape(depth, -1)

        sign_signal_buffer = back_sign_signal.reshape(depth,-1)

        indices = self.max_indices.reshape(depth, size, 2)

        for d in range(depth):

            output_slice = out_buffer[d]
            sign_signal_slice = sign_signal_buffer[d]
            indices_slice = indices[d]
            
            for k in range(size):
                
                sign_signal = sign_signal_slice[k]

                if sign_signal != 0 and output_slice[k] * sign_signal < 1.0:

                    i,j = indices_slice[k]

                    if i >= 0 and j >= 0:
                        forward_sign_signal[d][i][j] = sign_signal

        return forward_sign_signal


#############################################################################################################

#                               Standard alone helper functions                                             #
                            
#############################################################################################################

#Get shape size helper function
def shape_size(shape):

    s = len(shape)

    a = 1
        
    for i in range(s):
        a = a * shape[i]

    return a

#Flatten helper function
def flatten(x):
    return x.reshape(-1)

#Rectifier helper function
def rectify(x):
    np.maximum(x, 0, x)

def pad(x, padding):
        
    if padding > 0:
       return np.pad(x, ((0,0),(padding, padding),(padding, padding)),'constant')
    else:
       return x

# convert img to columns for vectorizing the convolution operation
def im2col(x, num_kernels, kernel_shape, stride, cols_buffer):

    kernel_height, kernel_width = kernel_shape
    
    size =  cols_buffer.shape[1]

    height_in = x.shape[1]
    width_in = x.shape[2]

    for j in range(num_kernels):

        col = 0
        row = 0

        for i in range(size):

           cols_buffer[:,i] = x[:,row:kernel_height+row, col:kernel_width + col].reshape(-1)
           
           col += stride

           if (kernel_width + col) > width_in:
                col = 0
                row += stride

           if (kernel_height + row) > height_in:
               break

def compute_output_dimension(input_shape, kernel_shape, stride, padding):

    kernel_height, kernel_width = kernel_shape
    
    depth, height_in, width_in = input_shape
    
    height_out = int((height_in - kernel_height + 2 * padding) / stride + 1)
    width_out = int((width_in - kernel_width + 2 * padding) / stride + 1)

    return height_out, width_out

#competitive credit assignment
def competitive_credit_assign(deltas_padded, trimming = 0):

    deltas = deltas_padded[:,trimming:deltas_padded.shape[1] - trimming,trimming:deltas_padded.shape[2] - trimming] #remove padding

    depth, height, width = deltas.shape
   
    channel_size = width * height
    
    deltas_buffer = deltas.reshape(depth, -1)

    sign_signal = np.zeros(deltas.shape)

    sign_signal_buffer = sign_signal.reshape(depth,-1)
    
    for i in range(channel_size):
        
        depth_column = deltas_buffer[:,i]

        assigned_channel = -1;

        max_delta = 0.0
        
        for n, delta in enumerate(depth_column):

             if abs(delta) > abs(max_delta):

                 max_delta = delta

                 assigned_channel = n
                 
        if assigned_channel > -1:

           sign_signal_buffer[assigned_channel][i] = np.sign(max_delta)

    return sign_signal

