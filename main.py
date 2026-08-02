# MLP implemented by Tyler Waltner 
# Date: Jul 10th, 2026
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
6	Shirt
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
        print("invalid input for activation func, please input either one of the following in the command line args for activation func:\nsig (sig OR sigmoid for Sigmoid, relu for ReLU, leaky_relu for Leaky ReLU, tanh for Tanh, smax for Soft Max)")
        sys.exit()


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
        print("invalid input for loss func, please input either one of the following in the command line args for loss:\nmse for Mean Squared Error, mae for Mean Absolute Error, bce for Binary Cross Entropy fl for Focal Loss")
        sys.exit()


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
        print("invalid input for feature scaling function, please input one of the following options: norm for Normalization, stdz for Standardization")
        sys.exit()


class MLP:
    def __init__(self, layers, actv_funcs, outer_func, loss):
        # weights and biases in 1D array, length of portions specified by input list input with our weights one row at a time product 
        # randomly generated based off of hidden layer dimensions, activation/loss funcs assigned by activ/loss func methods
        self.layers = layers
        self.actv_func, self.deriv_actv_func, self.cache = actv_funcs
        self.outer_func = outer_func
        self.loss_func, self.deriv_loss_func = loss
        
        # index dictionary to track indices of weights & biases of each layer
        self.indices = {"num_layers": len(layers) - 1} # (num_layers inclusive of output layer)

        # IF we are caching z values:
        if self.cache: 
            # cached_z does not need to store first layer (activations)
            self.cached_z = np.zeros(sum(self.layers[1:])) 

            # counter for tracking which z's haven't been cached yet
            self.cached_z_idx = 0

            # add indices of cached_z of each layer to self.indices (z values added later in forward pass)
            prev_idx = 0
            for i in range(1, self.indices["num_layers"] + 1):
                cur_idx = prev_idx + self.layers[i] - 1
                self.indices[f"layer_{i}_z"] = (prev_idx, cur_idx)
                prev_idx = cur_idx + 1

        # initialize array to store forward pass activations
        self.model_activations = np.zeros(sum(self.layers))
        
        # counter for tracking which activations haven't been populated inside of model_activations (utilized in forward pass)
        self.model_activations_idx = 0

        # add indices of model_activations of each layer to self.indices (activation values added later in forward pass)
        prev_idx = 0
        for i in range(0, self.indices["num_layers"] + 1):
            cur_idx = prev_idx + self.layers[i] - 1
            self.indices[f"layer_{i}_activations"] = (prev_idx, cur_idx)
            prev_idx = cur_idx + 1 # update prev_idx

        model_array = []
        prev_idx = 0    
        cur_idx = 0
        for i in range(1, len(layers)): # construct model_array and populate self.indices dictionary
            prev_layer = self.layers[i-1]
            
            cur_layer = self.layers[i]

            num_weights = prev_layer * cur_layer

            num_biases = cur_layer

            # initialize random weights & biases for layer
            model_array.append(np.random.randn(num_weights))
            model_array.append(np.zeros(num_biases))
            
            w_end_idx = prev_idx + num_weights - 1
            cur_idx = prev_idx + num_weights + num_biases - 1

            # update indices dict
            self.indices[f"layer_{i}_weights"] = (prev_idx, w_end_idx)
            self.indices[f"layer_{i}_biases"] = (w_end_idx + 1, cur_idx)

            # update prev_idx
            prev_idx = cur_idx + 1

        self.model_array = np.concatenate(model_array)


    def forward(self, activations: list[float], idx: int):
        if idx == 1: # if we are on first forward pass
            # reset idx tracker for activation
            self.model_activations_idx = 0

            if self.cache: # if caching, then also reset idx tracker for cached z values
                self.cached_z_idx = 0
            
            # add input layer's activations to self.model_activations
            num_a = len(activations)
            self.model_activations[self.model_activations_idx: num_a] = activations
            self.model_activations_idx +=  num_a # increment idx tracking

        # z = pre-activation value
        # a = activation
        # z = w * prev_a + b
        # cur_a = actv(z)

        # extract indexes
        w_start, w_end = self.indices[f"layer_{idx}_weights"]
        b_start, b_end = self.indices[f"layer_{idx}_biases"]

        # extract weights & biases from main array
        w_1d = self.model_array[w_start:w_end + 1]
        b_1d = self.model_array[b_start:b_end + 1]

        # transpose weights to 2d matrix to multiply by prev activations
        w_2d = w_1d.reshape(self.layers[idx], len(activations))
        
        # calulate pre-activation func
        cur_z = np.dot(w_2d, activations) + b_1d

        # IF caching z value:
        if self.cache:
            self.cached_z[self.cached_z_idx: self.cached_z_idx + len(cur_z)] = cur_z 
            self.cached_z_idx += len(cur_z) # increment idx tracking
    
        # pass z thru activation func
        cur_a = self.actv_func(cur_z)

        # append current activations (cur_a) to self.model_activations
        num_a = len(cur_a) # number of activations in layer
        self.model_activations[self.model_activations_idx: self.model_activations_idx + num_a] = cur_a
        self.model_activations_idx += num_a # increment idx tracking

        # recursion portion, if layer is NOT last layer (aka output layer): go deeper, else: return current activation
        if idx != self.indices["num_layers"]:
            print(f"Forward Pass of layer {idx} complete")
            return self.forward(cur_a, idx + 1)

        else:
            print(f"Forward Pass of layer {idx} complete")
            print("ending recursion")
            return cur_a # return output layer
        

    def backward(self, y, learning_rate: float, idx: int):
        # Backprop through layers
        # Chain Rule: 
        # F'(g(x)) = F'(g(x)) * g'(x)
        # z = w * a + b
        cur_layer = self.indices["num_layers"] + 1 - idx

        # extract tuples of indices for activations of current/prev layers, weights, and biases
        cur_a_beg, cur_a_end = self.indices[f"layer_{cur_layer}_activations"] 
        prev_a_beg, prev_a_end = self.indices[f"layer_{cur_layer - 1}_activations"] 
        w_beg, w_end = self.indices[f"layer_{cur_layer}_weights"]
        b_beg, b_end = self.indices[f"layer_{cur_layer}_biases"]

        # extract activations of current and prev layers from self.model_activations array and extract current weights/biases from self.model_array
        cur_a = self.model_activations[cur_a_beg: cur_a_end + 1] # +1 at end because list indexing is NOT inclusive of 2nd index
        prev_a = self.model_activations[prev_a_beg: prev_a_end + 1] # ^^^
        cur_w = self.model_array[w_beg: w_end + 1]                  # ^^^
        cur_b = self.model_array[b_beg: b_end + 1]                  # ^^^

        dC_da = None
        if idx == 1:
            dC_da = self.deriv_loss_func(y, cur_a)
        else: 
            dC_da = y

        # assign da/dz based off of if we cached or not
        da_dz = None
        if self.cache:
            cur_z_beg, cur_z_end = self.indices[f"layer_{cur_layer}_z"]
            cur_z = self.cached_z[cur_z_beg: cur_z_end + 1] # +1 at end because list indexing is NOT inclusive of 2nd index
            da_dz = self.deriv_actv_func(cur_z)
        else: # if we didn't cache, we can pass in current activation in deriv of activation func to obtain da/dz as activation func preserves gradients
            da_dz = self.deriv_actv_func(cur_a)

        # can use dC/db to calculate dC_dw since:
        # Change of Cost to biases: C = cost (aka loss), a = activ, z = value before activation b = bias
        # dC/db = C'(a(z(w))) * a'(z(w)) * 1
        # dC/db = C'(a(z(w))) * a'(z(w))
        # Change of Cost to Weights: C = cost (aka loss), a = activ, z = value before activation w = weight 
        # dC/dw = C'(a(z(w))) * a'(z(w)) * z'(w)
        # dC/dw = C'(a(z(w))) * a'(z(w)) * a(prev)
        # dC/dw = dC/db * a(prev)
        # calculate dC/db using element-wise mult 
        dC_db = dC_da * da_dz
        
        # use np.outer to use outer product generate 2d matrix from 1D vectors (breaks with np.dot as )
        dC_dw_2d = np.outer(dC_db, prev_a)
        dC_dw = dC_dw_2d.flatten() # flatten back to 1d array

        # backwards funciton needs to know which direction to decend, so we pass in dC/da(prev) to tell it
        # dC/da(prev) = dC/da * da/dz * dz/da(prev)
        #                               dz/da(prev) <= weights of cur layer (z = w * a + b)
        # dC/da(prev) = dC/da * da/dz * w(cur)
        # dC/da(prev) = dC/db * w(cur)
        # reshape weights to 2d matrix of dimensions (neurons current layer, neurons previous layer)
        dC_da_prev = np.dot(dC_db, cur_w.reshape(len(cur_a), len(prev_a)))

        # adjust weights / biases SUBTRACT= because want to DESCEND the gradient to minimize solution
        cur_w -= dC_dw * learning_rate
        cur_b -= dC_db * learning_rate

        # if not last layer, go deeper one more layer (else, is last layer and can end recursion)
        if idx != self.indices["num_layers"]: 
            print(f"Backward Pass of layer {idx} complete")

            # TODO CHANGE INPUTS FOR RECURSIVE STEP
            return self.backward(dC_da_prev, learning_rate, idx + 1)
        
        else:
            print(f"Backward Pass of layer {idx} complete")
            print("ending recursion")
    

# handle commandline input, organize information of model
def take_input():
    if len(sys.argv) < 10:
        print('Please input dataset you would like to learn:\nhidden layer size(s), num classes, choice of activation func for hidden layers, activation func for outer layers, loss func, epochs, and learning rate\nEx: python3 main.py data/fashion 128 16 10 sig smax mse 100 0.1 norm\ndataset: data/fashion\nhidden layers: 128, 16\nnum classes: 10\nactivation funcs: sig (sig OR sigmoid for Sigmoid, relu for ReLU, leaky_relu for Leaky Relu, smax for Soft Max, tanh for Tanh)\nloss func: mse (mse for Mean Squared Error, bce for Binary Cross Entropy fl for Focal Loss)\nepochs: 100\nlearning rate: 0.1, norm for normalization')
        sys.exit()

    data_path = sys.argv[1]
    hidden_layers = []
    num_hidden_layers = len(sys.argv) - 8
    
    for layer in range(2,num_hidden_layers):
        hidden_layers.append(int(sys.argv[layer]))

    num_classes = int(sys.argv[num_hidden_layers])
    actv_funcs = activation_func(sys.argv[num_hidden_layers + 1].lower()) # pass thru activation_func() to output correct actv_func 
    outer_func = activation_func(sys.argv[num_hidden_layers + 2].lower())
    loss = loss_func(sys.argv[num_hidden_layers + 3].lower())
    epochs = int(sys.argv[num_hidden_layers + 4])
    batch_size = int(sys.argv[num_hidden_layers + 5])
    learn_rate = float(sys.argv[num_hidden_layers + 6])
    feat_func = scaling_func(sys.argv[num_hidden_layers + 7])

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
    model = MLP(layers, actv_funcs, outer_func, loss)

    # training loop
    for e in range(epochs):
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
                
                model.forward(x, 1) # forward pass

                # construct output array to pass thru backward pass
                y_output = construct_label_array(y, num_classes)

                model.backward(y_output, learn_rate, 1) # backward pass


main()