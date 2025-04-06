import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as o
from model import CatDogClassificationModel
from dataset import Dataset
from model_image import ModelImage
from tqdm import tqdm

dataset = Dataset()
model_image = ModelImage(64)
model = CatDogClassificationModel(model_image.image_size)

# Use eval to initialize the model, because MLX uses lazy evaluation
mx.eval(model.parameters())


# Implement the loss function
# The loss function is a binary cross-entropy loss function
def loss_fn(model, input_data, expected):
    return nn.losses.binary_cross_entropy(model(input_data), expected, with_logits=True)


# Compute the gradients
vg = nn.value_and_grad(model, loss_fn)
optimizer = o.Adam(learning_rate=0.001)

# Training
batch_size = 16
training_dataset = dataset.get_training_dataset()[:10000]
training_results = []
epochs = 1

print(f"Training started.")

# Iterate over the dataset and train the model
for epoch in range(epochs):
    num_batches = len(training_dataset) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}"):
        batch = training_dataset.iloc[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        batch_images = []
        batch_labels = []

        for _, row in batch.iterrows():
            image_path = row["Image_Path"]
            label = row["Label"]

            loaded_image = model_image.load_image(image_path)
            batch_images.append(mx.array(loaded_image).reshape(1, -1))
            batch_labels.append(mx.array(label).reshape(1, 1))

        input_array = mx.array(batch_images).reshape(batch_size, -1)
        output_array = mx.array(batch_labels).reshape(batch_size, 1)
        loss, grads = vg(model, input_array, output_array)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        batch_loss = loss.item()
        training_results.append((epoch, batch_idx, batch_loss))

print(f"Training completed.")

model.save()
dataset.save_training_results(training_results)
dataset.plot_training_results(batch_size)
# print(model.show_parameters())
