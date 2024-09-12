import mlx.nn as nn
import numpy as np

# Neural network model for a simple calculator
# As input takes two numbers and an encoded calculator operation
# As an output returns a number


class NeuralNetwork(nn.Module):
    def __init__(self, exact_values=False):
        super().__init__()
        self.layers = [
            nn.Linear(3, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 1)
        ]
        self.exact_values = exact_values

    def __call__(self, x):
        x = self.layers[0](x)
        x = nn.relu(x)
        x = self.layers[1](x)
        x = nn.relu(x)
        x = self.layers[2](x)

        if self.exact_values:
            x = np.round(x).astype(int)

        return x

    def save(self):
        self.save_weights(self.model_path)

    def load(self):
        self.load_weights(self.model_path)

    def show_parameters(self):
        print("Model Parameters:")
        print(self.parameters())
