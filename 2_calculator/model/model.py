import mlx.core as mx
import os
from neural_network import NeuralNetwork


class CalculatorModel(NeuralNetwork):
    def __init__(self, exact_values=False):
        super().__init__(exact_values)
        self.folder_path = "./2_calculator/results"
        self.model_path = "./2_calculator/results/calculator_model.safetensors"
        os.makedirs(self.folder_path, exist_ok=True)

    def test(self, input_1, input_2, calc_operation, expected_output):
        result = self.inference(input_1, input_2, calc_operation)

        same = result == expected_output

        return result, same

    def inference(self, input_1, input_2, calc_operation):
        input_array = mx.array([input_1, input_2, calc_operation])

        result = self(input_array)
        result_value = result[0].item()

        return result_value
