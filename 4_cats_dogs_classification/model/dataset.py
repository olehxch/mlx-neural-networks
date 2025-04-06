import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import kagglehub
from skimage.transform import resize


class Dataset:
    def __init__(self):
        self.training_dataset_path = './4_cats_dogs_classification/data/training_dataset.csv'
        self.validation_dataset_path = './4_cats_dogs_classification/data/validation_dataset.csv'

        if not os.path.exists(self.training_dataset_path) or not os.path.exists(self.validation_dataset_path):
            print("Dataset files do not exist. Loading and splitting dataset...")
            self.load_dataset()
            self.transform_dataset_paths()
            self.split_dataset()
        else:
            print("Dataset files already exist. Loading dataset...")
        # self.show_images_as_matrix(self.training_dataset)

        self.training_results_path = './4_cats_dogs_classification/results/training_loss.csv'
        self.plot_training_loss_path = './4_cats_dogs_classification/results/training_loss.png'
        self.plot_validation_results_path = './4_cats_dogs_classification/results/validation_results.png'

    def load_dataset(self):
        # 1. Download the dataset from Kaggle
        # The dataset is downloaded to the "~/.cache/kagglehub" folder
        path = kagglehub.dataset_download(handle="bhavikjikadara/dog-and-cat-classification-dataset")

        print("Path to dataset files:", path)

        self.root_folder = os.path.join(path, "PetImages")
        self.cat_folder = os.path.join(self.root_folder, "Cat")
        self.dog_folder = os.path.join(self.root_folder, "Dog")

    def transform_dataset_paths(self):
        # 2. Transform the dataset paths to a list of tuples
        self.cat_images = os.listdir(self.cat_folder)
        self.dog_images = os.listdir(self.dog_folder)

        self.cat_images = [os.path.join(self.cat_folder, image) for image in self.cat_images]
        self.dog_images = [os.path.join(self.dog_folder, image) for image in self.dog_images]

        print(self.cat_images[:5], self.dog_folder[:5])

    def split_dataset(self):
        # 3. Split the dataset into training and validation sets
        # Labels: 0 - Cat, 1 - Dog
        # (image_path, label)
        self.all_dataset_size = len(self.cat_images) + len(self.dog_images)
        self.training_dataset_size = int(self.all_dataset_size * 0.8)
        self.validation_dataset_size = int(self.all_dataset_size * 0.2)

        cat_images = list(zip(self.cat_images, [0] * len(self.cat_images)))
        dog_images = list(zip(self.dog_images, [1] * len(self.dog_images)))
        dataset = cat_images + dog_images
        random.shuffle(dataset)

        # Split the dataset into training and validation sets
        self.training_dataset = dataset[:self.training_dataset_size]
        self.validation_dataset = dataset[self.training_dataset_size:
                                          self.training_dataset_size + self.validation_dataset_size]

        # Save the training dataset to a CSV file
        training_df = pd.DataFrame(self.training_dataset, columns=["Image_Path", "Label"])
        training_df.to_csv(self.training_dataset_path, index=False)

        # Save the validation dataset to a CSV file
        validation_df = pd.DataFrame(self.validation_dataset, columns=["Image_Path", "Label"])
        validation_df.to_csv(self.validation_dataset_path, index=False)

        return dataset

    def delete_corrupt_images(self):
        corrupted_images = ["6318.jpg"]

        for image_path in self.cat_images + self.dog_images:
            try:
                img = plt.imread(image_path)
                if img is None:
                    print(f"Corrupt image: {image_path}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                os.remove(image_path)
                print(f"Deleted corrupt image: {image_path}")
        print("Corrupt images deleted.")

    def load_images(self, images):
        images = images.astype(np.float32) / 255.0
        images = images.reshape(images.shape[0], -1)

        return images

    def load_labels(self, labels):
        labels = np.array([np.eye(10)[i] for i in labels])

        return labels

    def get_training_dataset(self):
        return pd.read_csv(self.training_dataset_path)

    def get_validation_dataset(self):
        return pd.read_csv(self.validation_dataset_path)

    def save_training_results(self, training_results):
        training_results_json = [{"Epoch": epoch, "Iteration": i+1, "Loss": loss} for epoch, i, loss in training_results]
        df = pd.DataFrame(training_results_json)
        df.to_csv(self.training_results_path, index=False, header=True)

    def plot_training_results(self, batch_size=None):
        csv_data = pd.read_csv(self.training_results_path)
        df = pd.DataFrame(csv_data, columns=["Epoch", "Iteration", "Loss"])

        plt.figure(figsize=(10, 6))
        for epoch, epoch_data in df.groupby("Epoch"):
            plt.plot(epoch_data["Iteration"], epoch_data["Loss"], label=f"Epoch {epoch + 1}")

        plt.xlabel(f"Batch Iteration (x{batch_size})")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.legend()
        plt.grid(True)

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

    def show_images_as_matrix(self, images, rows=5, cols=5):
        fig, axes = plt.subplots(rows, cols, figsize=(5, 5))
        images = images[:25]

        for i, image in enumerate(images):
            ax = axes[i // rows, i % cols]
            ax.imshow(plt.imread(image[0]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('on')

        fig.canvas.manager.set_window_title("Cat and Dog Images")
        plt.tight_layout()
        plt.subplots_adjust(top=1.0, wspace=0.1, hspace=0.1)  # Adjust space for the title
        plt.show()
