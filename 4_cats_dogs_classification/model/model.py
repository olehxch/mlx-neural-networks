import mlx.core as mx
import os

from neural_network import NeuralNetwork
from model_image import ModelImage

model_image = ModelImage(64)

class CatDogClassificationModel(NeuralNetwork):
    def __init__(self, image_size, exact_values=False):
        super().__init__(image_size, exact_values)
        self.folder_path = "./4_cats_dogs_classification/results"
        self.model_path = "./4_cats_dogs_classification/results/cats_docs_classification_model.safetensors"
        
        os.makedirs(self.folder_path, exist_ok=True)

    def inference(self, input):
        input_array = mx.array(input)
        result = self(input_array)

        return result

    def save(self):
        self.save_weights(self.model_path)

    def load(self):
        self.load_weights(self.model_path)

    def show_parameters(self):
        print("Model Parameters:")
        print(self.parameters())
