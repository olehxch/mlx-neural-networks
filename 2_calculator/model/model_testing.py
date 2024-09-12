from model import CalculatorModel
from dataset import Dataset

data = Dataset()

# Neural network returns very close values but not exact values
# to test the model with exact values, change the exact_values parameter to True
#
# To see the difference between exact and non-exact values, run the model with exact_values=False
model = CalculatorModel(exact_values=False)
model.load()

dataset = data.get_validation_dataset()

count_success = 0
count_failure = 0

for i, row in dataset.iterrows():
    input_1 = int(row.values[0])
    input_2 = int(row.values[1])
    calc_operation = data.encode_calc_operation(row.values[2])
    expected_output = int(row.values[3])

    model_output, same = model.test(input_1, input_2, calc_operation, expected_output)

    if same:
        count_success += 1
    else:
        count_failure += 1

    print(f"[{str(i+1).zfill(3)}] Input 1: {input_1}, Input 2: {input_2}, Expected Output: {expected_output}, Model Output: {model_output}, Same: {same}")

print(f"Success: {count_success}, Failure: {count_failure}")

data.plot_validation_results(count_success, count_failure)
