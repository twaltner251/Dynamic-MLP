# MLP implemented by Tyler Waltner 
# Date: Jun 26th, 2026
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

def activation_func(func: str):
    f = func.lower() # normalize str by making lowercase

    if f == 'sig' or f == 'sigmoid':
        def sigmoid(value: float):
            return 1 / (1 + math.pow(math.e, -value))
        return sigmoid
    
    elif f == 'relu':
        def relu(value: float):
            return 0 if value <= 0 else 1
        return relu
    
    elif f == 'tanh':
        def tanh(value: float):
            return np.tanh(value)
        return tanh
    
    else:
        print('invalid input for activation func, please input either one of the following in the commandline arguments:\nactivation func: sig (sig OR sigmoid for Sigmoid, relu for ReLU, tanh for Tanh)')
        sys.exit()


class MLP:
    def __init__(self, layers, actv_func):
        # weights and biases in 1D array, length of portions specified by input list input with our weights one row at a time product randomly generated based off of hidden layer dimensions, activation func assigned by activ func method
        self.layers = layers
        self.actv_func = actv_func
        self.indices = {"num_layers": len(layers) - 1} # index dictionary to track indices of weights & biases of each layer

        model_array = []
        prev_idx = 0    
        cur_idx = 0
        for i in range(1, len(layers)): # construct model_array and populate self.indices dictionary
            layer_name = f"layer_{i}"

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
            self.indices[f"{layer_name}_weights"] = (prev_idx, w_end_idx)
            self.indices[f"{layer_name}_biases"] = (w_end_idx + 1, cur_idx)

            # update prev_idx
            prev_idx = cur_idx + 1

        self.model_array = np.concatenate(model_array)

    def forward(self, activations: list[float], idx: int):
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

        # reshape weights into 2d array for matrix mult
        w_2d = w_1d.reshape(self.layers[idx], len(activations))
        
        # calulate pre-activation func
        z = np.dot(w_2d, activations) + b_1d
        
        # pass z thru activation func
        cur_a = self.actv_func(z)

        # recursion portion, if layer is NOT last layer: go deeper, else: return current activation
        if idx != self.indices["num_layers"]:
            print("Pass: " + str(idx) + "complete")
            return self.forward(cur_a, idx + 1)

        else:
            print("ending recursion")
            return cur_a

    def backward(self):
        pass
    

def take_input(): # handle commandline input, organize information of model
    if len(sys.argv) < 6:
        print('Please input dataset you would like to learn:\nhidden layer size(s), num classes, choice of activation func, epochs, and learning rate\nEx: python3 main.py data/fashion 128 16 10 sig 100 0.1\ndataset: data/fashion\nhidden layers: 128, 16\nnum classes: 10\nactivation func: sig (sig OR sigmoid for Sigmoid, relu for ReLU, tanh for Tanh)\nepochs: 100\nlearning rate: 0.1')
        return

    data_path = sys.argv[1]
    hidden_layers = []
    num_hidden_layers = len(sys.argv) - 4
    
    for layer in range(2,num_hidden_layers):
        hidden_layers.append(int(sys.argv[layer]))

    num_classes = int(sys.argv[num_hidden_layers])
    actv_func = activation_func(sys.argv[num_hidden_layers + 1]) # pass thru activation_func() to output correct actv_func 
    epochs = int(sys.argv[num_hidden_layers + 2])
    learn_rate = float(sys.argv[num_hidden_layers + 3])

    return data_path, hidden_layers, num_classes, actv_func, epochs, learn_rate


def main():
    data_path, hidden_layers, num_classes, actv_func, epochs, learn_rate = take_input()

    # load data
    X_train, y_train = mnist_reader.load_mnist(data_path, kind='train')
    X_test, y_test = mnist_reader.load_mnist(data_path, kind='t10k')
    # input_size = len(X_train[0])
    input_size = 2
    # CHANGE ^^^
    
    layers = [input_size] + hidden_layers + [num_classes]

    print(layers)

    model = MLP(layers, actv_func)

    print(model.forward([1,2], 1))


main()