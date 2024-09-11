import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as o
from xor_gate import XOR
from data import Data

# Instantiate the classes
data = Data()
xor = XOR()

# Use eval to initialize the model, because MLX uses lazy evaluation
mx.eval(xor.parameters())


# Implement the loss function
def loss_fn(model, input_data, expected):
    return mx.mean(mx.square(
        model(input_data) - expected
    ))


# Compute the gradients
vg = nn.value_and_grad(xor, loss_fn)
optimizer = o.SGD(learning_rate=0.01)

# Training
df = data.generate_test_dataset()
training_results = []

# Iterate over the dataset and train the model
for i, row in df.iterrows():
    input_1 = int(row.values[0])
    input_2 = int(row.values[1])
    output = int(row.values[2])

    input_array = mx.array([input_1, input_2])
    output_array = mx.array([output])
    loss, grads = vg(xor, input_array, output_array)

    optimizer.update(xor, grads)
    mx.eval(xor.parameters(), optimizer.state)
    training_results.append((i, loss.item()))

    if not i % 1000:
        print(f"Loss for '{i}': {loss.item()}")

# Save the trained model
xor.save()

# Save training results
data.save_training_results(training_results)

# Plot loss
data.plot_training_loss()

# Show parameters
# print(xor.show_parameters())
