import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as o
from model import CalculatorModel
from dataset import Dataset

# Instantiate the classes
dataset = Dataset()
model = CalculatorModel()

# Use eval to initialize the model, because MLX uses lazy evaluation
mx.eval(model.parameters())


# Implement the loss function
def loss_fn(model, input_data, expected):
    return nn.losses.mse_loss(model(input_data), expected)


# Compute the gradients
vg = nn.value_and_grad(model, loss_fn)
optimizer = o.Adam(learning_rate=0.01)

# Training
df = dataset.get_training_dataset()
training_results = []
epochs = 1

# Iterate over the dataset and train the model
for epoch in range(epochs):
    for i, row in df.iterrows():
        input_1 = int(row.values[0])
        input_2 = int(row.values[1])
        calc_operation = dataset.encode_calc_operation(row.values[2])
        output = int(row.values[3])

        input_array = mx.array([input_1, input_2, calc_operation])
        output_array = mx.array([output])
        loss, grads = vg(model, input_array, output_array)

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        training_results.append((i, loss.item()))

        if not i % 1000:
            print(f"Loss for '{i}': {loss.item()}")

    print(f"Epoch {epoch + 1} completed")

# Save the trained model
model.save()

# Save training results
dataset.save_training_results(training_results)

# Plot loss
dataset.plot_training_results()

# Show parameters
# print(model.show_parameters())
