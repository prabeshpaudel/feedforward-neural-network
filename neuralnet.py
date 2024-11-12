from utils import *
import numpy as np

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate=0.1, activation_function='sigmoid', l2_lambda=0.0, dropout_rate=0.0, weight_init='random'):
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes
        self.learning_rate = learningrate
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        # Set the activation function and its derivative for backpropagation
        if activation_function in activation_functions:
            self.activation_function = activation_functions[activation_function]
            self.activation_derivative = get_activation_derivative(activation_function)
        else:
            raise ValueError(f"Invalid activation function '{activation_function}'. Choose from {list(activation_functions.keys())}")

        # Initialize weights based on the chosen weight initialization method
        self.weight_input_hidden = self.initialize_weights(self.hidden_nodes, self.input_nodes, method=weight_init)
        self.weight_hidden_output = self.initialize_weights(self.output_nodes, self.hidden_nodes, method=weight_init)

    def initialize_weights(self, output_size, input_size, method='xavier'):
        """Initialize weights based on the chosen method."""
        if method == 'random':
            return np.random.rand(output_size, input_size) * 0.01  # Small random values
        elif method == 'xavier':
            return np.random.randn(output_size, input_size) * np.sqrt(1.0 / input_size)
        elif method == 'he':
            return np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        else:
            raise ValueError(f"Invalid weight initialization method '{method}'. Choose from 'random', 'xavier', or 'he'.")

    def train(self, inputs_list, target_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # Forward pass with optional dropout applied to hidden layer
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        if self.dropout_rate > 0:  # Apply dropout only if dropout_rate is specified
            dropout_mask = (np.random.rand(*hidden_outputs.shape) > self.dropout_rate).astype(float)
            hidden_outputs *= dropout_mask
            hidden_outputs /= (1 - self.dropout_rate)  # Scale to maintain expected output level

        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Error calculation and backpropagation
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors)

        final_grad = self.activation_derivative(final_outputs)
        hidden_grad = self.activation_derivative(hidden_outputs)

        # Update weights with optional L2 regularization
        if self.l2_lambda > 0:
            self.weight_hidden_output += self.learning_rate * (np.dot((output_errors * final_grad), np.transpose(hidden_outputs)) - self.l2_lambda * self.weight_hidden_output)
            self.weight_input_hidden += self.learning_rate * (np.dot((hidden_errors * hidden_grad), np.transpose(inputs)) - self.l2_lambda * self.weight_input_hidden)
        else:
            self.weight_hidden_output += self.learning_rate * np.dot((output_errors * final_grad), np.transpose(hidden_outputs))
            self.weight_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_grad), np.transpose(inputs))

    def predict(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # No dropout applied during prediction
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
