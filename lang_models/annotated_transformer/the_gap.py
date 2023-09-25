import numpy as np
import torch
import torch.nn as nn

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
