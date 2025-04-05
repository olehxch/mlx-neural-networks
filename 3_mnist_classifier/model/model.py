import mlx.core as mx
import os
from neural_network import NeuralNetwork


class NumberClassifierModel(NeuralNetwork):
    def __init__(self, exact_values=False):
        super().__init__(exact_values)
        self.folder_path = "./3_mnist_classifier/results"
        self.model_path = "./3_mnist_classifier/results/number_classifier_model.safetensors"
        os.makedirs(self.folder_path, exist_ok=True)

    def test(self, input, expected_output):
        result = self.inference(input)
        same = result == expected_output

        return result, same

    def inference(self, input):
        input_array = mx.array(input)
        result = self(input_array)
        result_value = mx.argmax(result).item()

        return result_value
