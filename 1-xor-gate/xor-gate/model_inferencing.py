from xor_gate import XOR
from data import Data

data = Data()
xor = XOR()
xor.load()

# Inference for a single input

model_output = xor.inference(1, 1)
print(f"Model output for 1, 1: {model_output}")
