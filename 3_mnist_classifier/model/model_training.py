import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as o
from model import NumberClassifierModel
from dataset import Dataset

# Instantiate the classes
dataset = Dataset()
model = NumberClassifierModel()

# Use eval to initialize the model, because MLX uses lazy evaluation
mx.eval(model.parameters())


# Implement the loss function
def loss_fn(model, input_data, expected):
    return nn.losses.mse_loss(model(input_data), expected)


# Compute the gradients
vg = nn.value_and_grad(model, loss_fn)
optimizer = o.Adam(learning_rate=0.01)

# Training
images, labels = dataset.get_training_dataset()
training_results = []
epochs = 3

# Iterate over the dataset and train the model
for epoch in range(epochs):
    i = 0

    for image, label in zip(images, labels):
        input_array = mx.array(mx.array(image))
        output_array = mx.array(label)
        loss, grads = vg(model, input_array, output_array)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        training_results.append((i, loss.item()))

        if not i % 1000:
            print(f"Loss for '{i}': {loss.item()}")

        i += 1

    print(f"Epoch {epoch + 1} completed")

# Save the trained model
model.save()

# Save training results
dataset.save_training_results(training_results)

# Plot loss
dataset.plot_training_results()

# Show parameters
# print(model.show_parameters())
