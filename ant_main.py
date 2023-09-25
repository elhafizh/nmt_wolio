from lang_models.annotated_transformer import the_gap
from lang_models.annotated_transformer import building_block as ant_bb
import torch

"""
The Annotated Transformer module
"""

if __name__ == "__main__":

    """The GAP: SimpleNN"""
    # Define the input data (a batch of 3 samples, each with 2 features)
    input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Create an instance of the SimpleNN model (input_size=2, output_size=1)
    model = the_gap.SimpleNN(input_size=2, output_size=1)

    # Print the model's parameters (weights and biases)
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter value: {param}")

    # Apply the model to the input data
    output = model(input_data)

    # Print the output
    print("Output:")
    print(output)
