import mlx.core as mx
from model import CatDogClassificationModel
from dataset import Dataset
from model_image import ModelImage
from tqdm import tqdm

data = Dataset()
model_image = ModelImage(64)
model = CatDogClassificationModel(model_image.image_size, True)
model.load()

validation_dataset = data.get_validation_dataset()

count_success = 0
count_failure = 0
model_outputs = []

for i, row in tqdm(validation_dataset.iterrows(), total=len(validation_dataset), desc="Validation Progress"):
    image_path = row["Image_Path"]
    label = row["Label"]

    loaded_image = model_image.load_image(image_path)
    input_array = mx.array(loaded_image).reshape(1, -1)
    expected_output = mx.array(label).reshape(1, 1)

    model_output = model.inference(input_array)
    same = model_output == expected_output

    model_outputs.append(model_output)
    count_success += same.item()
    count_failure += not same

print(f"Success N: {count_success}, Failure: {count_failure}")
print(f"Success %: {round(count_success / len(validation_dataset) * 100, 2)}, Failure: {round(count_failure / len(validation_dataset) * 100, 2)}")
# data.plot_validation_results(count_success, count_failure)
