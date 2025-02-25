# Import pprint, module we use for making our print statements prettier
import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch

from lang_models.annotated_transformer import building_block as ant_bb
from lang_models.annotated_transformer import the_gap

pp = pprint.PrettyPrinter()

"""
The Annotated Transformer module
"""


def main_simpleNN():
    # Define the input data (a batch of 3 samples, each with 2 features)
    input_data = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    # Create an instance of the SimpleNN model (input_size=2, output_size=1)
    model = the_gap.SimpleNN(input_size=2, output_size=1)

    # Save linear pytorch parameters for further utilizaton using numpy
    parameters = {"weights": 0, "bias": 0}

    # Print the model's parameters (weights and biases)
    for name, param in model.named_parameters():
        if name == "linear.weight":
            parameters["weights"] = param.detach().numpy()
        if name == "linear.bias":
            parameters["bias"] = param.detach().numpy()
        # print(f"Parameter name: {name}")
        # print(f"Parameter value: {param}")

    # Apply the model to the input data
    output = model(input_data)

    # Print the output
    print("Output of SimpleNN:")
    print(output)

    print("------------------------------------------------------------")
    print("parameters :")
    pp.pprint(parameters)
    print("------------------------------------------------------------")

    """The GAP: linear_transformation()"""

    # Define input data (3 samples, each with 2 features)
    input_data = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    print(input_data.shape)

    # Define weights (2 input features, 1 output feature)
    weights = np.array([[0.5], [1.0]])  # replace with parameters["weights"]

    # Define bias (1 output feature)
    bias = np.array([0.5])  # replace with parameters["bias"]

    # Perform the linear transformation
    output = the_gap.linear_transformation(
        input_data=input_data,
        weights=parameters["weights"].transpose(),
        bias=parameters["bias"],
    )

    # Print the output
    print("Output of linear_transformation():")
    print(output)


def main_tologsoftmax():
    logits = torch.tensor([2.0, 1.0, 0.1])
    the_gap.to_logsoftmax_predict(logits=logits)
    print("------------------------------------------------------------")
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.1],
            [0.5, 1.5, 2.5],
        ]
    )
    the_gap.to_logsoftmax_predict(logits=logits)


def main_matmulNNparams():
    # create an instance model
    model = the_gap.matmulWithNNParams()

    # Print the model's parameters
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter value:\n{param}\n")

    # Generate some input data
    input_data = torch.randn(2, 3)

    # Forward pass through the model
    output = model(input_data)

    # Print the output
    print("Output:")
    print(output)


def main_NNDropout():
    # Create an instance of MyModel
    model = the_gap.SimpleNNDropout()

    # Generate some input data
    input_data = torch.randn(3, 10)  # Batch size of 3, input size of 10

    # Forward pass through the model
    output = model(input_data)

    # Print the output
    print("Output:")
    print(output)


if __name__ == "__main__":
    """The GAP: nn.Linear"""
    main_simpleNN()

    """The GAP: nn.functional.log_softmax"""
    main_tologsoftmax()

    """The GAP: nn.Parameters"""
    main_matmulNNparams()

    """The GAP: visualize uniform/normal distribution"""
    the_gap.generate_visualize_dist("normal", 1000)

    """The GAP: visualize 2D Tensor"""
    tensor = torch.randn(3, 6)
    the_gap.visualize_2D_tensor(tensor=tensor)

    """The GAP: simple neural network with dropout"""
    main_NNDropout()
