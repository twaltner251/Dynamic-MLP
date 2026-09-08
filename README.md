# Dynamic Multi-Layer Perceptron (MLP) from Scratch

**Author:** Tyler Waltner

**Date:** August 27th, 2026

**Contact:** [waltnertyler@gmail.com](mailto:waltnertyler@gmail.com) | [twaltner@u.rochester.edu](mailto:twaltner@u.rochester.edu)

## 🧠 Project Overview

A fully custom, from-scratch Multi-Layer Perceptron (MLP) built in Python and NumPy to classify the FashionMNIST dataset.

I purposely engineered this project without relying on high-level machine learning frameworks (like PyTorch or TensorFlow) to build a foundational, first-hand understanding of Deep Learning. By working entirely from scratch, I was able to directly apply my university coursework in linear algebra, calculus, and artificial intelligence to design low-level memory optimizations and custom backpropagation algorithms.

## ✨ Key Technical Achievements

* **Memory-Optimized Architecture:** Engineered a unique 1D array state system for weights and biases. By pre-allocating `np.zeros` arrays and modifying them in-place via index slicing during recursive forward and backward passes, I successfully eliminated NumPy's memory reallocation overhead.
* **Algorithmic Efficiency (Calculus Optimization):** Streamlined backpropagation by calculating the gradient with respect to biases first, then using it to efficiently derive the gradient with respect to weights, preventing redundant computations.
* **Dynamic & Configurable:** Highly flexible architecture driven by command-line arguments. Supports dynamic hidden layer sizes, dynamic feature scaling (normalization/standardization), and interchangeable activation and loss functions.
* **Smart State Caching:** Built a custom activation function handler that selectively caches pre-activation values *only* when required by functions that destroy gradients (e.g., ReLU), highly optimizing overall RAM usage.
* **Custom Error Pipeline:** Incorporated smooth error pipelining to safely isolate and handle user input errors at runtime.

## 🧮 Mathematical Optimization (Backpropagation)

Instead of recalculating the full chain rule for both weights and biases, the backward pass propagates through layers recursively. By isolating the bias gradient $\frac{\partial C}{\partial b}$, we can compute the weight gradient $\frac{\partial C}{\partial w}$ with a simple multiplication, saving significant compute time:

$$\frac{\partial C}{\partial b} = C'(a(z)) \cdot a'(z)$$

$$\frac{\partial C}{\partial w} = C'(a(z)) \cdot a'(z) \cdot \frac{\partial z}{\partial w}$$

$$\frac{\partial C}{\partial w} = \frac{\partial C}{\partial b} \cdot a_{prev}$$

*(Where $C$ is cost/loss, $a$ is the activation, $z$ is the pre-activation value, $b$ is bias, and $w$ is weight).*

## 🛠️ Notable Challenges Solved

* **Bypassing Immutable Memory Allocation:** Initially, I planned to append each layer's activations dynamically using `np.append`. However, because NumPy arrays are immutable in size, this would create an entirely new array in memory every time. To optimize memory, I allocated an `np.zeros` array scaled to the exact size of all hidden layers combined, and utilized a `self.model_activations_idx` tracker to modify the array strictly in-place during the forward pass. This same optimization was applied to the `self.cached_z` array.
* **Lightweight Batch Shuffling:** Instead of using `random.shuffle(dataset)` (which duplicates a massive matrix in memory), I implemented a lightweight index array (e.g., `[0, 1, ..., 59999]`). I shuffle this 1D array to pull batch indices, drastically reducing memory consumption during epoch training.

## 🚀 Instructions for Use

Run the model via the command line by specifying your desired hyperparameters:

```bash
# Syntax:
python3 main.py <dataset> <hidden_layers...> <num_classes> <hidden_activ> <out_activ> <loss_func> <epochs> <learning_rate> <scaling>

# Example:
python3 main.py data/fashion 128 16 10 sig smax mse 100 0.1 norm

```

**Parameter Breakdown:**

* **dataset:** Path to dataset (e.g., `data/fashion`)
* **hidden layers:** Space-separated node counts (e.g., `128 16`)
* **num classes:** Number of output classifications (e.g., `10`)
* **activation funcs:**
* `sig` or `sigmoid` (Sigmoid)
* `relu` (ReLU)
* `leaky_relu` (Leaky ReLU)
* `smax` (Softmax)
* `tanh` (Tanh)


* **loss func:** `mse` (Mean Squared Error), `bce` (Binary Cross Entropy), or `fl` (Focal Loss)
* **epochs:** Integer (e.g., `100`)
* **learning rate:** Float (e.g., `0.1`)
* **scaling:** `norm` (Normalization) or `std` (Standardization)

## 🗺️ Roadmap (Version 2 TODOs)

* [ ] Complete custom training loop to pass data in true batches.
* [ ] Implement specific train vs. testing dataset parsing (skip dataset loads if testing a saved model).
* [ ] Add model state saving/loading (via `.txt` or `.npy`) to preserve training progress.
* [ ] Build an analytical performance method to track F1 score & loss over time.
* [ ] Finalize derivatives for BCE & Focal Loss functions.

## 📝 Changelog

* Added feature scaling (`scaling_func()` for normalization vs standardization) via CLI.
* Added Softmax activation function.
* Added CLI parameters for output layer activation and batch size.
* Setup data dataloader shuffling in the `main()` training loop.

* Backwards pass implemented successfully; corrected vector math.
* Removed redundant `.flatten()` operations for element-wise products.
* Fixed Tanh derivative (switched from $1 - \exp(arr, 2)$ to standard $np.square(arr)$ relation).
* Fixed ReLU derivative using `np.where(vector > 0, 1, 0)` instead of `np.clip` to handle binary gradient mapping correctly.
* Added Leaky ReLU to combat exploding gradients observed during testing.

* Fixed `self.indices` to correctly include the first input layer in `model.activations`.
* Updated `activation_func()` to return a boolean dictating whether state caching is required.
* Implemented forward pass caching array initialization (`self.cached_z_values`) that only activates if caching is needed (skipping the input layer).
* Vectorized activation functions to process full NumPy arrays simultaneously rather than iterating iteratively.
* Switched entirely to index slicing instead of appending to optimize memory overhead.

* Added loss function parameters to CLI (`mse`, `mae`, `bce`, `focal loss`).
* Updated forward pass to save state activations in `self.model_activations` for recursive backpropagation.
* Fixed output layer bug where weight and bias matrices were skipping instantiation in `self.indices`.

---

*Note on AI Use: AI was utilized strictly as an educational resource to grasp complex ML concepts (e.g., the chain rule calculus for backpropagation, F1/loss metrics, and general 1D architecture theory), and for minor syntax debugging. The core algorithms, memory optimization logic, and Python implementation were written entirely from scratch.*
