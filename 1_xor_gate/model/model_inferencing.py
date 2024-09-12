from model import XOR
from dataset import Dataset

data = Dataset()
xor = XOR()
xor.load()

model_output = xor.inference(1, 1)
print(f"Model output for 1, 1: {model_output}")
