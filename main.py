from neuralnet import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# Define learning rates for different activation functions
learning_rates = {
    "relu": 0.001,
    "sigmoid": 0.1,
    "tanh": 0.01,
    "leaky_relu": 0.001
}

# List of activation functions and weight initialization methods to test
activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
weight_initializations = ['random', 'xavier', 'he']


# Set parameters for the neural network
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
epochs = 5  # Reduced number of epochs for quicker execution during testing

# Load the MNIST dataset from TensorFlow
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten the images into vectors
train_images = train_images.reshape(train_images.shape[0], 784)
test_images = test_images.reshape(test_images.shape[0], 784)


# Function to normalize images based on activation function requirements
def normalize_images(images, activation_function):
    if activation_function == 'sigmoid':
        # Scale to [0.01, 1.0] for sigmoid activation
        return images / 255.0 * 0.99 + 0.01
    elif activation_function == 'tanh':
        # Scale to [-1.0, 1.0] for tanh activation
        return (images / 255.0 - 0.5) * 2
    elif activation_function in ['relu', 'leaky_relu']:
        # Scale to [0.0, 1.0] for ReLU and Leaky ReLU activation
        return images / 255.0
    else:
        return images


# Function to calculate AUC for multi-class classification
def compute_auc(model, test_images, test_labels):
    all_true_labels = np.zeros((len(test_labels), 10))  # Assuming 10 classes in MNIST
    all_predictions = np.zeros_like(all_true_labels)

    # Convert test_labels to one-hot encoding
    for i, label in enumerate(test_labels):
        all_true_labels[i, label] = 1

    for i in range(len(test_images)):
        inputs = test_images[i]
        outputs = model.predict(inputs)
        all_predictions[i] = outputs.T  # Transpose to match dimensions

    # Compute AUC for each class and average them
    auc_score = roc_auc_score(all_true_labels, all_predictions, average='macro', multi_class='ovr')

    return auc_score


# Visualize a sample from the MNIST training dataset
index = 0  # specify which sample to display
image_array = train_images[index].reshape((28, 28))
# Uncomment the following lines to display the image
# plt.imshow(image_array, cmap='Greys', interpolation='none')
# plt.show()
print("The target value is:", train_labels[index])

# Dictionary to store performance, loss results, and AUC results for each combination
performance_results = {}
loss_results = {}
auc_results = {}

# Loop over each activation function and weight initialization method
for weight_init in weight_initializations:
    for activation_function in activation_functions:
        print(f"\nTraining with activation function: {activation_function} and weight initialization: {weight_init}")

        # Get the corresponding learning rate
        learning_rate = learning_rates[activation_function]

        # Normalize images according to the activation function
        train_inputs = normalize_images(train_images, activation_function)
        test_inputs = normalize_images(test_images, activation_function)

        # Create a neural network instance with the specified activation function and weight initialization method
        nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, activation_function,
                                 weight_init=weight_init)

        # Track metrics during training
        epoch_losses = []
        epoch_auc_scores = []

        # Train the neural network over multiple epochs
        for e in range(epochs):
            print("Epoch:", e + 1, "/", epochs)
            epoch_loss = 0
            for i in range(len(train_inputs)):
                inputs = train_inputs[i]
                targets = np.zeros(output_nodes) + 0.01
                targets[train_labels[i]] = 0.99
                # Train the model and accumulate loss
                loss = nn_model.train(inputs, targets)
                epoch_loss += loss
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / len(train_inputs)
            epoch_losses.append(avg_epoch_loss)
            print(f"Average loss for epoch {e + 1}: {avg_epoch_loss:.4f}")

            # Calculate AUC after each epoch
            auc_score = compute_auc(nn_model, test_inputs, test_labels)
            epoch_auc_scores.append(auc_score)
            print(f"AUC after epoch {e + 1}: {auc_score:.4f}")

        # Store the loss history and AUC scores for analysis
        loss_results[(activation_function, weight_init)] = epoch_losses
        auc_results[(activation_function, weight_init)] = epoch_auc_scores

        # Evaluate the network's performance on the test set
        scorecard = []
        for i in range(len(test_inputs)):
            correct_label = test_labels[i]
            inputs = test_inputs[i]
            outputs = nn_model.predict(inputs)
            label = np.argmax(outputs)
            # Append 1 if correct, 0 if incorrect
            scorecard.append(1 if label == correct_label else 0)

        # Calculate and print performance
        scorecard_array = np.asarray(scorecard)
        performance = (scorecard_array.sum() / scorecard_array.size) * 100
        print(f"Performance with {activation_function} activation and {weight_init} initialization: {performance:.2f}%")

        # Store performance in the results dictionary
        performance_results[(activation_function, weight_init)] = performance

sorted_results = sorted(performance_results.items(), key=lambda x: x[1], reverse=True)
best_config = sorted_results[0]  # Highest performance
worst_config = sorted_results[-1]  # Lowest performance

print("\nBest configuration:", best_config)
print("Worst configuration:", worst_config)

regularizations = {
    "l2_lambda": [0.001, 0.01, 0.1],
    "dropout_rate": [0.2, 0.4, 0.6]
}

reg_performance_results = {}
reg_loss_results = {}
reg_auc_results = {}


# Function to run regularization tests on fixed activation function and weight initialization
def run_regularization_tests(activation_function, weight_init, config_name):
    print(f"\nTesting regularizations for {config_name} configuration: {activation_function} and {weight_init}")
    for reg_type, reg_values in regularizations.items():
        for reg_value in reg_values:
            print(f"\nTraining with {reg_type}={reg_value}")

            # Get the corresponding learning rate
            learning_rate = learning_rates[activation_function]

            # Normalize images according to the activation function
            train_inputs = normalize_images(train_images, activation_function)
            test_inputs = normalize_images(test_images, activation_function)

            # Create a neural network instance with regularization
            if reg_type == "l2_lambda":
                nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,
                                         activation_function, weight_init=weight_init, l2_lambda=reg_value,
                                         dropout_rate=0)
            elif reg_type == "dropout_rate":
                nn_model = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,
                                         activation_function, weight_init=weight_init, l2_lambda=0,
                                         dropout_rate=reg_value)
            else:
                return ValueError(f"Invalid regularization")

            # Train the neural network
            epoch_losses = []
            epoch_auc_scores = []
            for e in range(epochs):
                print("Epoch:", e + 1, "/", epochs)
                epoch_loss = 0
                for i in range(len(train_inputs)):
                    inputs = train_inputs[i]
                    targets = np.zeros(output_nodes) + 0.01
                    targets[train_labels[i]] = 0.99
                    # Train the model and accumulate loss
                    loss = nn_model.train(inputs, targets)
                    epoch_loss += loss
                # Average loss for the epoch
                avg_epoch_loss = epoch_loss / len(train_inputs)
                epoch_losses.append(avg_epoch_loss)
                print(f"Average loss for epoch {e + 1}: {avg_epoch_loss:.4f}")

                # Calculate AUC after each epoch
                auc_score = compute_auc(nn_model, test_inputs, test_labels)
                epoch_auc_scores.append(auc_score)
                print(f"AUC after epoch {e + 1}: {auc_score:.4f}")

            # Evaluate performance
            scorecard = []
            for i in range(len(test_inputs)):
                correct_label = test_labels[i]
                inputs = test_inputs[i]
                outputs = nn_model.predict(inputs)
                label = np.argmax(outputs)
                scorecard.append(1 if label == correct_label else 0)

            performance = (np.asarray(scorecard).sum() / len(test_inputs)) * 100
            print(f"Performance with {reg_type}={reg_value}: {performance:.2f}%")

            # Store results for plotting and analysis
            key = (activation_function, weight_init, reg_type, reg_value)
            reg_loss_results[key] = epoch_losses
            reg_auc_results[key] = epoch_auc_scores
            reg_performance_results[key] = performance


# Regularization tests for best and worst configurations
best_activation, best_weight_init = best_config[0]
worst_activation, worst_weight_init = worst_config[0]

run_regularization_tests(best_activation, best_weight_init, "Best")
run_regularization_tests(worst_activation, worst_weight_init, "Worst")

# Results of Performance, Average Loss, AUC Score

# Plot the training loss for each combination
plt.figure(figsize=(12, 8))
for (activation_function, weight_init), losses in loss_results.items():
    plt.plot(range(1, epochs + 1), losses, label=f"{activation_function} - {weight_init}")
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss for Different Activation Functions and Weight Initialization Methods')
plt.legend()
plt.show()

# Plot the AUC scores for each combination
plt.figure(figsize=(12, 8))
for (activation_function, weight_init), auc_scores in auc_results.items():
    plt.plot(range(1, epochs + 1), auc_scores, label=f"{activation_function} - {weight_init}")
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.title('AUC Scores for Different Activation Functions and Weight Initialization Methods')
plt.legend()
plt.show()

# Print out a summary of performance for all combinations of activation functions and weight initializations
print("\nSummary of performance for different activation functions and weight initializations:")
for (activation_function, weight_init), performance in performance_results.items():
    print(f"{activation_function} with {weight_init} initialization: {performance:.2f}%")


# Regularization Results
plt.figure(figsize=(12, 8))
for (activation_function, weight_init, reg_type, reg_value), losses in reg_loss_results.items():
    if (activation_function, weight_init) in [(best_activation, best_weight_init),
                                              (worst_activation, worst_weight_init)]:
        plt.plot(range(1, epochs + 1), losses, label=f"{activation_function}-{weight_init}, {reg_type}={reg_value}")
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss with Regularization')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
for (activation_function, weight_init, reg_type, reg_value), auc_scores in reg_auc_results.items():
    if (activation_function, weight_init) in [(best_activation, best_weight_init),
                                              (worst_activation, worst_weight_init)]:
        plt.plot(range(1, epochs + 1), auc_scores, label=f"{activation_function}-{weight_init}, {reg_type}={reg_value}")
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.title('AUC Scores with Regularization')
plt.legend()
plt.show()

print("\nSummary of performance for different activation functions, weight initializations, and regularizations:")
for (activation_function, weight_init, reg_type, reg_value), performance in reg_performance_results.items():
    print(f"{activation_function} with {weight_init} initialization and {reg_type}={reg_value}: {performance:.2f}%")
