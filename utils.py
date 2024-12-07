import numpy as np

# Define activation functions with clipping to prevent overflow
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    x = np.clip(x, -500, 500)
    return np.tanh(x)

def softmax(x):
    # Shift values to avoid large exponents (helps prevent overflow)
    x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)

# Dictionary of activation functions
activation_functions = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh,
    'softmax': softmax,
    'leaky_relu': leaky_relu
}

def get_activation_derivative(activation_function):
    # Return the derivative function based on the activation function name
    if activation_function == 'sigmoid':
        # Derivative of sigmoid function, where x is the output of sigmoid(x)
        return lambda x: x * (1.0 - x)
    elif activation_function == 'tanh':
        # Derivative of tanh function, where x is the output of tanh(x)
        return lambda x: 1 - np.square(x)
    elif activation_function == 'relu':
        # Derivative of ReLU function
        return lambda x: (x > 0).astype(float)
    elif activation_function == 'leaky_relu':
        # Derivative of Leaky ReLU function
        return lambda x: np.where(x > 0, 1, 0.01)
    else:
        raise ValueError("Unsupported activation function for derivatives")
