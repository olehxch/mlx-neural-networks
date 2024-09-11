import mlx.core as mx
import mlx.nn as nn
import numpy as np
import os

# Neural network model for XOR gate
# Input -> Hidden -> Output
# O -> O \
#         -> O
# O -> O /
#
# As input takes two numbers, e.g. [0, 1]
# As an output returns one number, e.g. [1]


class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(2, 2),
            nn.Linear(2, 1)
        ]
        self.folder_path = "./1-xor-gate/results"
        self.model_path = "./1-xor-gate/results/xor_model.safetensors"
        os.makedirs(self.folder_path, exist_ok=True)

    def __call__(self, x):
        x = self.layers[0](x)
        x = nn.tanh(x)
        x = self.layers[1](x)
        return x

    def test(self, input_1: int, input_2: int, expected_output: int):
        input_array = mx.array([input_1, input_2])

        result = self(input_array)
        result_round = int(np.round(result).item())
        same = result_round == expected_output

        return result_round, same

    def inference(self, input_1: int, input_2: int):
        input_array = mx.array([input_1, input_2])

        result = self(input_array)
        result_round = int(np.round(result).item())

        return result_round

    def save(self):
        self.save_weights(self.model_path)

    def load(self):
        self.load_weights(self.model_path)

    def show_parameters(self):
        print("Model Parameters:")
        print(self.parameters())
