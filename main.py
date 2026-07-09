# MLP implemented by Tyler Waltner 
# Date: Jul 5th, 2026
# Email: waltnertyler@gmail.com

import math
import numpy as np
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

# returns tuple of activation func as well as its derivative to be passed into MLP
def activation_func(func: str): # last bool in output tuple represents whether caching is needed or not in Forward pass
    f = func.lower() # normalize str by making lowercase

    if f == 'sig' or f == 'sigmoid':
        def sigmoid(vector):
            return 1 / (1 + np.exp(-vector))
        def deriv_sigmoid(previous_sigmoid): 
            # forward pass calculated our sigmoid values, so when we input we can use it to calculate derivative of sig(x) which = (1 - sig(x)) * sig(x)
            return (1 - previous_sigmoid) * previous_sigmoid
        return sigmoid, deriv_sigmoid, False
    
    elif f == 'relu':
        def relu(vector):
            return np.maximum(0, vector)
        def deriv_relu(vector):
            return np.clip(vector, 0, 1) 
        return relu, deriv_relu, True # relu need to cache due to destroying gradients when value is < 0
    
    elif f == 'tanh':
        def tanh(vector):
            return np.tanh(vector)
        def deriv_tanh(previous_tanh):
            return 1 - np.exp(previous_tanh, 2)
        return tanh, deriv_tanh, False
    
    else:
        print("invalid input for activation func, please input either one of the following in the command line args for activation func:\nsig (sig OR sigmoid for Sigmoid, relu for ReLU, tanh for Tanh)")
        sys.exit()


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


class MLP:
    def __init__(self, layers, actv_funcs, loss):
        # weights and biases in 1D array, length of portions specified by input list input with our weights one row at a time product 
        # randomly generated based off of hidden layer dimensions, activation/loss funcs assigned by activ/loss func methods
        self.layers = layers
        self.actv_func, self.deriv_actv_func, self.cache = actv_funcs
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
        if idx == 1: # if we are on first forward pass, add in input layer's activations
            for a in activations:
                self.model_activations[self.model_activations_idx] = a
                self.model_activations_idx += 1 # increment idx tracking

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

        # append each activation in cur_a (current activations) to self.model_activations
        for a in cur_a:
            self.model_activations[self.model_activations_idx] = a
            self.model_activations_idx += 1 # increment idx tracking

        # recursion portion, if layer is NOT last layer (aka output layer): go deeper, else: return current activation
        if idx != self.indices["num_layers"]:
            print(f"Forward Pass of layer {idx} complete")
            return self.forward(cur_a, idx + 1)

        else:
            print(f"Forward Pass of layer {idx} complete")
            print("ending recursion")
        

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
        cur_w = self.model_array[w_beg: w_end + 1]                      # ^^^
        cur_b = self.model_array[b_beg: b_end]                          # ^^^

        dC_da = None
        if idx == 1:
            dC_da = self.deriv_loss_func(y, cur_a)
        else: 
            dC_da = y

        print("y:", np.shape(y))

        print("dC_da:", np.shape(dC_da))

        da_dz = None
        if self.cache:
            cur_z_beg, cur_z_end = self.indices[f"layer_{cur_layer}_z"]
            cur_z = self.cached_z[cur_z_beg: cur_z_end + 1] # +1 at end because list indexing is NOT inclusive of 2nd index
            da_dz = self.deriv_actv_func(cur_z)
        else:
            da_dz = self.deriv_actv_func(cur_a)
        
        print("da_dz:", np.shape(da_dz))

        # can use dC/db to calculate dC_dw since:
        # Change of Cost to biases: C = cost (aka loss), a = activ, z = value before activation b = bias
        # dC/db = C'(a(z(w))) * a'(z(w)) * 1
        # dC/db = C'(a(z(w))) * a'(z(w))
        # Change of Cost to Weights: C = cost (aka loss), a = activ, z = value before activation w = weight 
        # dC/dw = C'(a(z(w))) * a'(z(w)) * z'(w)
        # dC/dw = C'(a(z(w))) * a'(z(w)) * a(prev)
        # dC/dw = dC/db * a(prev)
        dC_db = np.dot(dC_da, da_dz)    
        
        print("dC/db = C'(a(z(w))) * a'(z(w)) * 1:", np.shape(dC_db))
        
        dC_dw = np.dot(dC_db, prev_a)

        print("dC/dw = dC/db * a(prev):", np.shape(dC_dw))
    
        # adjust weights / biases SUBTRACT= because want to DESCEND the gradient to minimize solution
        cur_w -= dC_dw * learning_rate

        print("cur_w:", np.shape(cur_w))

        cur_b -= dC_db * learning_rate

        print("cur_b:", np.shape(cur_b))

        # backwards funciton needs to know which direction to decend, so we pass in dC/da(prev) to tell it
        # dC/da(prev) = dC/da * da/dz * dz/da(prev)
        #                               dz/da(prev) <= weights of cur layer (z = w * a + b)
        # dC/da(prev) = dC/da * da/dz * w(cur)
        # dC/da(prev) = dC/db * w(cur)
        dC_da_prev = np.dot(dC_db, cur_w)

        print("dC_da_prev:", np.shape(dC_da_prev))

        # if not last layer, go deeper one more layer (else, is last layer and can end recursion)
        if idx != self.indices["num_layers"]: 
            print(f"Backward Pass of layer {idx} complete")

            # TODO CHANGE INPUTS FOR RECURSIVE STEP
            return self.backward(dC_da_prev, learning_rate, idx + 1)
        
        else:
            print(f"Backward Pass of layer {idx} complete")
            print("ending recursion")
    

def take_input(): # handle commandline input, organize information of model
    if len(sys.argv) < 7:
        print('Please input dataset you would like to learn:\nhidden layer size(s), num classes, choice of activation func, loss func, epochs, and learning rate\nEx: python3 main.py data/fashion 128 16 10 sig mse 100 0.1\ndataset: data/fashion\nhidden layers: 128, 16\nnum classes: 10\nactivation func: sig (sig OR sigmoid for Sigmoid, relu for ReLU, tanh for Tanh)\nloss func: mse (mse for Mean Squared Error, bce for Binary Cross Entropy fl for Focal Loss)\nepochs: 100\nlearning rate: 0.1')
        return

    data_path = sys.argv[1]
    hidden_layers = []
    num_hidden_layers = len(sys.argv) - 5
    
    for layer in range(2,num_hidden_layers):
        hidden_layers.append(int(sys.argv[layer]))

    num_classes = int(sys.argv[num_hidden_layers])
    actv_funcs = activation_func(sys.argv[num_hidden_layers + 1].lower()) # pass thru activation_func() to output correct actv_func 
    loss = loss_func(sys.argv[num_hidden_layers + 2].lower())
    epochs = int(sys.argv[num_hidden_layers + 3])
    learn_rate = float(sys.argv[num_hidden_layers + 4])

    return data_path, hidden_layers, num_classes, actv_funcs, loss, epochs, learn_rate


def main():
    # handle command line args
    data_path, hidden_layers, num_classes, actv_funcs, loss, epochs, learn_rate = take_input()

    # load data
    X_train, y_train = mnist_reader.load_mnist(data_path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')
    
    input_size = len(X_train[0]) # set input size
    
    layers = [input_size] + hidden_layers + [num_classes]

    # change back to layers
    #
    #
    model = MLP([3,2,1], actv_funcs, loss)

    # change to first input
    #
    #
    model.forward([0, 0, 1], 1)
    
    y = [1]
    model.backward([1], learn_rate, 1)


main()