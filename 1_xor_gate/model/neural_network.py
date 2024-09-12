import mlx.nn as nn

# Neural network model for XOR gate
# Input -> Hidden -> Output
# O -> O \
#         -> O
# O -> O /
#
# As input takes two numbers, e.g. [0, 1]
# As an output returns one number, e.g. [1]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(2, 2),
            nn.Linear(2, 1)
        ]

    def __call__(self, x):
        x = self.layers[0](x)
        x = nn.tanh(x)
        x = self.layers[1](x)

        return x

    def save(self):
        self.save_weights(self.model_path)

    def load(self):
        self.load_weights(self.model_path)

    def show_parameters(self):
        print("Model Parameters:")
        print(self.parameters())
