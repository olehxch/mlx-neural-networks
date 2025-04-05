import mlx.nn as nn
import numpy as np

# Neural network model for a NIST handwritten number classification
# As input takes 768 pixels of an image with the number
# As an output returns a number in the array with the position from 0 to 9
# that represents the number


class NeuralNetwork(nn.Module):
    def __init__(self, exact_values=False):
        super().__init__()
        self.layers = [
            nn.Linear(784, 40),
            nn.Linear(40, 10)
        ]
        self.exact_values = exact_values

    def __call__(self, x):
        x = self.layers[0](x)
        x = nn.sigmoid(x)
        x = self.layers[1](x)

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
