import mlx.core as mx
from model import NumberClassifierModel
from dataset import Dataset

data = Dataset()

# Neural network returns very close values but not exact values
# to test the model with exact values, change the exact_values parameter to True
# To see the difference between exact and non-exact values, run the model with exact_values=False
model = NumberClassifierModel(exact_values=False)
model.load()

images, labels = data.get_validation_dataset()

i = 0
count_success = 0
count_failure = 0

for image, label in zip(images, labels):
    input_array = mx.array(mx.array(image))
    expected_output = label

    model_output, same = model.test(input_array, expected_output)

    if same:
        count_success += 1
    else:
        count_failure += 1

    print(f"[{str(i+1).zfill(3)}] Expected Output: {expected_output}, Model Output: {model_output}, Same: {same}")
    i += 1

print(f"Success N: {count_success}, Failure: {count_failure}")
print(f"Success %: {count_success / len(images) * 100}, Failure: {count_failure / len(images) * 100}")

data.plot_validation_results(count_success, count_failure)
