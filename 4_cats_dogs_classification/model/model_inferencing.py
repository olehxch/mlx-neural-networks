from model import CalculatorModel
from dataset import Dataset

data = Dataset()
model = CalculatorModel(exact_values=True)
model.load()

# Inference for a single input
input_1 = 1
input_2 = 3
calc_operation = "+"
calc_operation_encoded = data.encode_calc_operation(calc_operation)

model_output = model.inference(input_1, input_2, calc_operation_encoded)
print(f"Model output for '{input_1} {calc_operation} {input_2} = {model_output}'")
