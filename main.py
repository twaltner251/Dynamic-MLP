# MLP implemented by Tyler Waltner 
# Date: Aug 27th, 2026
# Emails: waltnertyler@gmail.com
#         twaltner@u.rochester.edu

import numpy as np
import random
import sys
import mnist_reader

# data organized as 60,000 rows of 784 uint8 (unsigned 8bit ints) ranging from 0 to 255, representing grayscale darkness
'''
Labels Array:
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirts
7	Sneaker
8	Bag
9	Ankle boot
'''

 
# returns tuple of activation func, its derivative, and bool represents whether caching is needed or not in Forward pass to be passed into MLP
def activation_func(func: str): 
    f = func.lower() # normalize str by making lowercase

    if f == 'sig' or f == 'sigmoid':
        def sigmoid(vector):
            return 1 / (1 + np.exp(-vector))
        def deriv_sigmoid(prev_sig): 
            # forward pass calculated our sigmoid values, so when we input we can use it to calculate derivative of sig(x) which = (1 - sig(x)) * sig(x)
            return (1 - prev_sig) * prev_sig
        return sigmoid, deriv_sigmoid, False
   
    elif f == 'tanh':
        def tanh(vector):
            return np.tanh(vector)
        def deriv_tanh(prev_tanh):
            return 1 - np.square(prev_tanh)
        return tanh, deriv_tanh, False
    
    elif f == 'relu':
        def relu(vector):
            return np.maximum(0, vector)
        def deriv_relu(vector):
            return np.where(vector > 0, 1, 0) # where vector > 0, assign value to 1, else = 0
        return relu, deriv_relu, True # relu need to cache due to destroying gradients when value is < 0
    
    elif f == 'leaky_relu':
        def leaky_relu(vector):
            return np.maximum(0.01 * vector, vector)
        def deriv_relu(vector):
            return np.where(vector > 0, 1, 0.01) # where vector > 0, assign value to 1, else = 0
        return leaky_relu, deriv_relu, True # relu need to cache due to destroying gradients when value is < 0
    
    elif f == 'smax': # soft max
        def soft_max(vector):
            e_exp = np.exp(vector)
            return e_exp / np.sum(e_exp)
        def deriv_soft_max(prev_smax):
            # deriv of softmax is jacobian
            return np.diag(prev_smax) - np.outer(prev_smax, prev_smax)
        return soft_max, deriv_soft_max, True

    else:
        raise TypeError("invalid input for activation func, please input either one of the following in the command line args for activation func:\nsig (sig OR sigmoid for Sigmoid, relu for ReLU, leaky_relu for Leaky ReLU, tanh for Tanh, smax for Soft Max)")


# returns tuple of loss func as well as derivative to be passed into MLP
def loss_func(func: str):
    # y = actual, y_hat = predicted
    if func == "mse":
        # wrap outputs in float to convert from np.float64 -> float
        def mean_sq_err(y: list[float], y_hat: list[float]):
            return float(np.mean((y_hat - y) ** 2))
        def deriv_mean_sq_err(y: list[float], y_hat: list[float]):
            return (2 / len(y)) * (y_hat - y)
        return mean_sq_err, deriv_mean_sq_err
    
    elif func == "mae":
        def mean_abs_err(y: list[float], y_hat: list[float]):
            return float(np.mean(np.abs(y - y_hat)))
        def deriv_mean_abs_err(y: list[float], y_hat: list[float]):
            return np.sign(y - y_hat) / len(y)
        return mean_abs_err, deriv_mean_abs_err
        
    elif func == "bce":
        def binary_cross_entropy(y: list[float], y_hat: list[float]):
            return 
        def deriv_binary_cross_entropy(y: list[float], y_hat: list[float]):
            return
        return binary_cross_entropy, deriv_binary_cross_entropy
    
    elif func == "fl":
        def focal_loss(y: list[float], y_hat: list[float]):
            return
        def deriv_focal_loss(y: list[float], y_hat: list[float]):
            return
        return focal_loss, deriv_focal_loss
    
    else:
        raise TypeError("invalid input for loss func, please input either one of the following in the command line args for loss:\nmse for Mean Squared Error, mae for Mean Absolute Error, bce for Binary Cross Entropy fl for Focal Loss")
        

# return appropriate feature scaling function
def scaling_func(func: str):
    if func == "norm":
        def normalize(vector):
            return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
        return normalize

    elif func == "stdz":
        def standardize(vector):
            return (vector - np.mean(vector)) / np.std(vector)
        return standardize

    else:
        raise TypeError("invalid input for feature scaling function, please input one of the following options: norm for Normalization, stdz for Standardization")


class MLP:
    def __init__(self, layers, actv_funcs, outer_funcs, loss, batch_size):
        self.layers = layers # input + hidden + output layer #'s in array
        self.a_func, self.da_func, self.a_cache = actv_funcs # self.cache is boolean of if we cache for this function or not
        self.out_a_func, self.out_da_func, self.out_a_cache = outer_funcs 
        self.l_func, self.dl_func = loss
        self.batch_size = batch_size = batch_size
        self.cache = []

        # arrays to store weights & biases
        self.weights = [] 
        self.biases = []

        for i in range(1, len(self.layers)):
            # print(i, layers[i])
            self.weights.append(np.random.randn(layers[i], layers[i - 1]))
            if i != len(self.layers) - 1: # if not on last layer append to biases
                self.biases.append(np.zeros(layers[i]))


    def forward(self, prev_a: list[float], idx: int):
        # z = pre-activation value
        # a = activation
        # z = w * prev_a + b
        # cur_a = actv(z)
        w = self.weights[idx].T
        b = self.biases[idx]

        print('idx', idx)
        print('prev a', np.shape(prev_a))
        print('w', np.shape(w))
        print('b', np.shape(b))
        
        z = np.dot(prev_a, w) + b

        if idx != len(self.layers) - 2: # if not last pass
            if self.a_cache: # if caching
                self.cache.append(z) # cache z
        
            cur_a = self.a_func(z) # pass pre-activation value thru activation func

        else: # if last pass, use last activation func
            if self.out_a_cache: # if caching
                self.cache.append(z) # cache z

            cur_a = self.out_a_func(z) # pass pre-activation value thru activation func

        idx += 1 # increment idx

        if idx != len(self.layers) - 2: # if not on last pass, go deeper
            return self.forward(cur_a, idx)
        
        else: # if last pass
            return cur_a # return output activation


    def backward(self, y, learning_rate: float, idx: int):
        # Backprop through layers
        # Chain Rule: 
        # F'(g(x)) = F'(g(x)) * g'(x)
        # z = w * a + b
        
        # can use dC/db to calculate dC_dw since:
        # Change of Cost to biases: C = cost (aka loss), a = activ, z = value before activation b = bias
        # dC/db = C'(a(z(w))) * a'(z(w)) * 1
        # dC/db = C'(a(z(w))) * a'(z(w))
        # Change of Cost to Weights: C = cost (aka loss), a = activ, z = value before activation w = weight 
        # dC/dw = C'(a(z(w))) * a'(z(w)) * z'(w)
        # dC/dw = C'(a(z(w))) * a'(z(w)) * a(prev)
        # dC/dw = dC/db * a(prev)
        # calculate dC/db using element-wise mult 
        
        # backwards funciton needs to know which direction to decend, so we pass in dC/da(prev) to tell it
        # dC/da(prev) = dC/da * da/dz * dz/da(prev)
        #                               dz/da(prev) <= weights of cur layer (z = w * a + b)
        # dC/da(prev) = dC/da * da/dz * w(cur)
        # dC/da(prev) = dC/db * w(cur)
        # reshape weights to 2d matrix of dimensions (neurons current layer, neurons previous layer)

        return

# handle commandline input, organize information of model
def take_input():
    example_input = "python3 main.py data/fashion 128 16 10 sig smax mse 100 64 0.1 norm\n" \
                    "dataset: data/fashion\n" \
                    "hidden layers: 128, 16\n" \
                    "num classes: 10\n" \
                    "inner activation func: sig (sig OR sigmoid for Sigmoid, relu for ReLU, leaky_relu for Leaky Relu, smax for Soft Max, tanh for Tanh)\n" \
                    "loss func: mse (mse for Mean Squared Error, bce for Binary Cross Entropy fl for Focal Loss)\n" \
                    "outer activation func: smax (same as above...)\n" \
                    "loss: mse (mse for Mean Squared Error, mae for Mean Absolute Error, bce for Binary Cross Entropy fl for Focal Loss)" \
                    "epochs: 100\n" \
                    "batch size: 64\n" \
                    "learning rate: 0.1\n" \
                    "feature scaling: norm (norm for Normalization, stdz for Standardization)"

    num_hidden_layers = max(0, len(sys.argv) - 10) # can't have num_hidden_layers be negative
    
    # So if len(sys.argv) - hidden_layers < 10, you guaranteed don't have enough inputs
    if len(sys.argv) - num_hidden_layers < 10: 
        print(f'Not enough inputs, please see example input: {example_input}')
        sys.exit()
    
    data_path = sys.argv[1]
    hidden_layers = []
    
    for layer in sys.argv[2: 2 + num_hidden_layers]:
        hidden_layers.append(int(layer))

    try:
        num_classes = int(sys.argv[num_hidden_layers + 2])
        # print('classes', num_classes)
    except ValueError as e:
        print(f"Error parsing number of classes: {e}.\nReceived: {sys.argv[num_hidden_layers + 2]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        actv_funcs = activation_func(sys.argv[num_hidden_layers + 3].lower()) # pass thru activation_func() to output correct actv_func 
        # print('inner', actv_funcs)
    except Exception as e:
        print(f"Error parsing activation function for hidden layers: {e}.\nReceived: {sys.argv[num_hidden_layers + 3]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        outer_func = activation_func(sys.argv[num_hidden_layers + 4].lower())
        # print('outer', outer_func)
    except Exception as e:
        print(f"Error parsing activation function for outer layers: {e}.\nReceived: {sys.argv[num_hidden_layers + 4]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        loss = loss_func(sys.argv[num_hidden_layers + 5].lower())
        # print('loss', loss)
    except Exception as e:
        print(f"Error parsing loss function: {e}.\nReceived: {sys.argv[num_hidden_layers + 5]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        epochs = int(sys.argv[num_hidden_layers + 6])
        # print('epoch', epochs)
    except ValueError as e:
        print(f"Error parsing number of epochs: {e}.\nReceived: {sys.argv[num_hidden_layers + 6]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        batch_size = int(sys.argv[num_hidden_layers + 7])
        # print('batch:', batch_size)
    except ValueError as e:
        print(f"Error parsing batch size: {e}.\nReceived: {sys.argv[num_hidden_layers + 7]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        learn_rate = float(sys.argv[num_hidden_layers + 8])
        # print('learn', learn_rate)
    except ValueError as e:
        print(f"Error parsing learning rate: {e}.\nReceived: {sys.argv[num_hidden_layers + 8]}\n\nExample input: {example_input}")
        sys.exit()

    try:
        feat_func = scaling_func(sys.argv[num_hidden_layers + 9])
        # print('feat:', feat_func)
    except Exception as e:
        print(f"Error parsing feature scaling function: {e}.\nReceived: {sys.argv[num_hidden_layers + 9]}\n\nExample input: {example_input}")
        sys.exit()

    return data_path, hidden_layers, num_classes, actv_funcs, outer_func, loss, epochs, batch_size, learn_rate, feat_func


# returns "true" output layer to be compared to predicted
def construct_label_array(y_label: int, output_size: int): 
    label_arr = np.zeros(output_size)
    label_arr[y_label] = 1
    return label_arr


def main():
    # handle command line args
    data_path, hidden_layers, num_classes, actv_funcs, outer_func, loss, epochs, batch_size, learn_rate, feat_func = take_input()

    # load data
    X_train, y_train = mnist_reader.load_mnist(data_path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')
    
    # normalize data
    X_train = feat_func(X_train)
    X_test = feat_func(X_test)

    input_size = len(X_train[0]) # set input size
    
    # idx array, [0, 1, 2, 3, ... , 59999]
    idx = np.arange(len(X_train))

    # construct layer count to be passed into MLP
    layers = [input_size] + hidden_layers + [num_classes]

    # instantiate model
    model = MLP(layers, actv_funcs, outer_func, loss, batch_size)

    test_model = MLP([3] + hidden_layers + [num_classes], actv_funcs, outer_func, loss, batch_size)
    test_model.forward(np.random.randn(3), 0)

    # training loop
    for e in range(epochs):
        print('pausing before training loop')
        return 
        # shuffle idx array in-place with np.random.shuffle() each epoch
        np.random.shuffle(idx)
    
        # iterates thru entire dataset using interval batch_size
        for k in range(0, len(X_train), batch_size):
            batch_idx = idx[k: k + batch_size]

            # iterate thru batch
            for i in batch_idx:
                # grab inputs and appropriate labels
                x = X_train[i]
                y = y_train[i]
                
                model.forward(x, 0) # forward pass

                # construct output array to pass thru backward pass
                y_output = construct_label_array(y, num_classes)

                model.backward(y_output, learn_rate, 0) # backward pass


main()