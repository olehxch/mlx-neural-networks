import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

mnist_files = np.load('./3_mnist_classification/data/mnist.npz')

# Load the MNIST dataset
x_train = mnist_files['x_train']
y_train = mnist_files['y_train']
x_test = mnist_files['x_test']
y_test = mnist_files['y_test']


class Dataset:
    def __init__(self):
        self.training_dataset_size = 20000
        self.validation_dataset_size = int(self.training_dataset_size * 0.2)

        self.training_dataset_path = './3_mnist_classification/data/training_dataset.csv'
        self.validation_dataset_path = './3_mnist_classification/data/validation_dataset.csv'

        self.training_results_path = './3_mnist_classification/results/training_loss.csv'
        self.plot_training_loss_path = './3_mnist_classification/results/training_loss.png'
        self.plot_validation_results_path = './3_mnist_classification/results/validation_results.png'

    def load_images(self, images):
        images = images.astype(np.float32) / 255.0
        images = images.reshape(images.shape[0], -1)

        return images

    def load_labels(self, labels):
        labels = np.array([np.eye(10)[i] for i in labels])

        return labels

    def get_training_dataset(self):
        images = self.load_images(x_train)
        labels = self.load_labels(y_train)

        return images, labels

    def get_validation_dataset(self):
        images = self.load_images(x_test)
        labels = [i.item() for i in y_test]

        return images, labels

    def save_training_results(self, training_results):
        training_results_json = [{"Iteration": i+1, "Loss": loss} for i, loss in training_results]
        df_test_cases_1bit = pd.DataFrame(training_results_json)
        df_test_cases_1bit.to_csv(self.training_results_path, index=False, header=True)

    def plot_training_results(self):
        csv_data = pd.read_csv(self.training_results_path)
        df = pd.DataFrame(csv_data, columns=["Iteration", "Loss"])
        plt.plot(df["Iteration"], df["Loss"])

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")

        plt.savefig(self.plot_training_loss_path)
        plt.show()

    def plot_validation_results(self, count_success, count_failure):
        labels = ['Success', 'Failed']
        x = [count_success, count_failure]
        colors = ['#00FF00', '#FF0000']

        _, ax1 = plt.subplots()

        ax1.pie(x=x, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')

        plt.title("Validation Results")
        plt.savefig(self.plot_validation_results_path)
        # plt.show()

    def show_image(self, index=0):
        image = x_test[index]  # Select the image at the given index
        plt.imshow(image, cmap='gray')  # Display the image in grayscale
        plt.title(f"Label: {y_test[index]}")  # Display the corresponding label
        plt.axis('off')  # Turn off the axis
        plt.show()

    def show_images_as_matrix(self, rows=10, cols=10):
        images, labels = self.get_training_dataset()
        images = images.reshape(-1, 28, 28)  # Reshape flattened images back to 28x28

        fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
        # fig.suptitle("MNIST Images", fontsize=14)

        for i in range(rows * cols):
            row, col = divmod(i, cols)
            axes[row, col].imshow(images[i], cmap='gray')
            # axes[row, col].set_title(f"Label: {labels[i]}")
            axes[row, col].axis('off')

        fig.canvas.manager.set_window_title("MNIST Visualization")
        plt.tight_layout()
        plt.subplots_adjust(top=1.0, wspace=0.1, hspace=0.1)  # Adjust space for the title
        plt.show()
