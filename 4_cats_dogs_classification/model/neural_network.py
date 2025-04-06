import mlx.nn as nn


# Neural network model for a cat and dog classification
# By default as an input takes 64x64 (4096) raw numbers that represent a flattened image
# The model is a simple feedforward neural network with one hidden layer
# The model is trained to classify images of cats (0) and dogs (1)
# As an output returns a number between 0 and 1
# The model is trained using binary cross-entropy loss function and Adam optimizer


class NeuralNetwork(nn.Module):
    def __init__(self, image_size, exact_values=False):
        super().__init__()
        self.image_size = image_size
        self.input_layer_size = self.image_size * self.image_size
        self.layers = [
            nn.Linear(self.input_layer_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ]
        self.exact_values = exact_values

    def __call__(self, x):
        for layers in self.layers:
            x = layers(x)

        # Apply sigmoid activation function for inference
        if self.exact_values:
            x = nn.sigmoid(x)
            x = x.flatten().item()
            x = 1 if (x >= 0.5) else 0

        return x
