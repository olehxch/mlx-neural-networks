from model import XOR
from dataset import Dataset

data = Dataset()
xor = XOR()
xor.load()

# Load JSON file
dataset = data.get_validation_dataset()

count_success = 0
count_failure = 0

# Iterate through each item in the JSON file
for i, row in dataset.iterrows():
    input_1 = int(row.values[0])
    input_2 = int(row.values[1])
    expected_output = int(row.values[2])

    model_output, same = xor.test(input_1, input_2, expected_output)

    if same:
        count_success += 1
    else:
        count_failure += 1

    print(f"[{str(i+1).zfill(3)}] Input 1: {input_1}, Input 2: {input_2}, XOR Output: {expected_output}, Model Output: {model_output}, Same: {same}")

print(f"Success: {count_success}, Failure: {count_failure}")

data.plot_validation_results(count_success, count_failure)
