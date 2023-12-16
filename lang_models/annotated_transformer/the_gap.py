import pprint
from typing import Annotated, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

pp = pprint.PrettyPrinter()

"""
nn.Linear : defining linear (fully connected) layers in a neural network
"""


# Define a simple neural network with one linear layer
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Apply the linear transformation
        output = self.linear(x)
        return output


# Define a simple linear transformation function
def linear_transformation(input_data, weights, bias):
    """
    input_data: Input data as a NumPy array (shape: [batch_size, input_size])
    weights: Weight matrix as a NumPy array (shape: [input_size, output_size])
    bias: Bias vector as a NumPy array (shape: [output_size])

    Returns the result of the linear transformation as a NumPy array (shape: [batch_size, output_size]).
    """
    # Perform the weighted sum
    weighted_sum = np.dot(input_data, weights)

    # Add the bias
    output = weighted_sum + bias

    return output


import torch.nn.functional as F
from torch import Tensor


def to_logsoftmax_predict(logits: Tensor) -> None:
    t_shape = logits.shape

    if len(t_shape) == 1:
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=0)
    else:
        # Apply log_softmax along the last dimension (dimension -1)
        log_probs = F.log_softmax(logits, dim=-1)

    # Print the log probabilities
    print("Log Probabilities:")
    print(log_probs)

    if len(t_shape) == 1:
        # Sum of log probabilities should be close to 1 (within numerical precision)
        print(
            "Sum of Log Probabilities (should be close to 1):",
            torch.sum(torch.exp(log_probs)),
        )
        # To get the predicted class, you can find the index with the highest log probability
        predicted_class = torch.argmax(log_probs).item()
        print("Predicted Class:", predicted_class)
    else:
        # Sum of log probabilities along dimension -1 should be close to 1
        print(
            "Sum of Log Probabilities along dimension -1 (should be close to 1):",
            torch.sum(torch.exp(log_probs), dim=-1),
        )
        # To get the predicted class for each sample, find the index with the highest log probability
        predicted_classes = torch.argmax(log_probs, dim=-1)
        print("Predicted Classes for Each Sample:")
        print(predicted_classes)


# Define a custom neural network module that utilizing nn.Parameter
class matmulWithNNParams(nn.Module):
    def __init__(self):
        super(matmulWithNNParams, self).__init__()
        # Create a learnable parameter
        self.weight = nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        # Use the learnable parameter in the forward pass
        output = torch.matmul(x, self.weight)
        return output


def generate_visualize_dist(
    dist_type: Literal["uniform", "normal"],
    qty: Annotated[int, "How much data do you want to generate?"],
) -> None:
    if dist_type == "uniform":
        # Generate random numbers between 0 and 1 from a uniform distribution
        data = torch.rand(qty)
    if dist_type == "normal":
        # Generate 1000 random numbers from a normal distribution with mean 0 and standard deviation 1
        data = torch.randn(qty)
    # Plot a histogram to visualize the distribution
    plt.hist(data, bins=20, density=True)
    plt.title(f"{dist_type} distribution")
    plt.xlabel("value")
    plt.ylabel("probability density")
    plt.show()


def visualize_thin_tensor(tensor: Tensor) -> None:
    if len(tensor.shape) == 2:
        pp.pprint(tensor)
        # Create a heatmap to visualize the tensor
        plt.imshow(tensor, cmap="viridis", aspect="auto")
        plt.colorbar(label="Value")
        # Access the individual dimension values
        dimensions = tensor.shape
        plt.title(f"Tensor ({dimensions[0]}x{dimensions[1]})")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.show()
    elif len(tensor.shape) == 3:
        # Iterate through the 3D tensor and plot each 2D slice
        for i in range(tensor.size(0)):
            plt.figure(figsize=(5, 4))  # Set the figure size
            plt.imshow(tensor[i], cmap='viridis', aspect='auto')
            plt.colorbar(label='Value')
            plt.title(f'Layer {i+1}')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.show()
    else:
        print("The tensor's shape must equal two or dimensions.")


# Define a simple neural network with dropout
class SimpleNNDropout(nn.Module):
    def __init__(self):
        super(SimpleNNDropout, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # Fully connected layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.fc2 = nn.Linear(5, 2)  # Another fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout with p=0.5
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Define the input data (a batch of 3 samples, each with 2 features)
    input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Create an instance of the SimpleNN model (input_size=2, output_size=1)
    model = SimpleNN(input_size=2, output_size=1)

    # Print the model's parameters (weights and biases)
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param}")

    # Apply the model to the input data
    output = model(input_data)

    # Print the output
    print("Output:")
    print(output)
