import random
import pandas as pd
import matplotlib.pyplot as plt
import os


class Dataset:
    def __init__(self):
        self.training_dataset_size = 20000
        self.validation_dataset_size = int(self.training_dataset_size * 0.2)

        self.training_dataset_path = './2_calculator/data/training_dataset.csv'
        self.validation_dataset_path = './2_calculator/data/validation_dataset.csv'

        self.training_results_path = './2_calculator/results/training_loss.csv'
        self.plot_training_loss_path = './2_calculator/results/training_loss.png'
        self.plot_validation_results_path = './2_calculator/results/validation_results.png'

        self.calc_operations = ['+', '-']

    def encode_calc_operation(self, operation):
        if operation == '+':
            return 0
        elif operation == '-':
            return 1

    def decode_calc_operation(self, operation):
        if operation == 0:
            return '+'
        elif operation == 1:
            return '-'

    def get_training_dataset(self):
        if not os.path.exists(self.training_dataset_path):
            return self.generate_training_dataset()
        else:
            return pd.read_csv(self.training_dataset_path)

    def get_validation_dataset(self):
        if not os.path.exists(self.validation_dataset_path):
            return self.generate_validation_dataset()
        else:
            return pd.read_csv(self.validation_dataset_path)

    def generate_training_dataset(self):
        test_cases = []

        for _ in range(self.training_dataset_size):
            input_1 = random.randint(0, 9)
            input_2 = random.randint(0, 9)
            calc_operation = random.choice(self.calc_operations)

            output = eval(f"{input_1} {calc_operation} {input_2}")
            test_cases.append({
                "Input 1": input_1,
                "Input 2": input_2,
                "Operation": calc_operation,
                "Expected output": output
            })

        df_test_cases = pd.DataFrame(test_cases)
        df_test_cases.to_csv(self.training_dataset_path, index=False, header=True)

        return df_test_cases

    def generate_validation_dataset(self):
        test_cases = []

        for _ in range(self.validation_dataset_size):
            input_1 = random.randint(0, 9)
            input_2 = random.randint(0, 9)
            calc_operation = random.choice(self.calc_operations)

            output = eval(f"{input_1} {calc_operation} {input_2}")
            test_cases.append({
                "Input 1": input_1,
                "Input 2": input_2,
                "Operation": calc_operation,
                "Expected output": output
            })

        df_test_cases = pd.DataFrame(test_cases)
        df_test_cases.to_csv(self.validation_dataset_path, index=False, header=True)

        return df_test_cases

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
