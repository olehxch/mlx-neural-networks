import mlx.core as mx
import numpy as np
import os
from neural_network import NeuralNetwork


class XOR(NeuralNetwork):
    def __init__(self):
        super().__init__()
        self.folder_path = "./1_xor_gate/results"
        self.model_path = "./1_xor_gate/results/xor_model.safetensors"
        os.makedirs(self.folder_path, exist_ok=True)

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
