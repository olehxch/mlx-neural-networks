import random
import pandas as pd
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.test_cases_1bit_xor_path = './1-xor-gate/data/test_cases_1bit_xor.csv'
        self.training_results_path = './1-xor-gate/results/training_results_loss.csv'
        self.plot_training_loss_path = './1-xor-gate/results/training_loss.png'
        self.plot_testing_results_path = './1-xor-gate/results/testing_results.png'

    def generate_test_dataset(self):
        test_cases_1bit = []

        for _ in range(10000):
            input_1 = random.randint(0, 1)
            input_2 = random.randint(0, 1)
            output = input_1 ^ input_2
            test_cases_1bit.append({
                "Input 1": input_1,
                "Input 2": input_2,
                "Expected output": output
            })

        df_test_cases_1bit = pd.DataFrame(test_cases_1bit)
        df_test_cases_1bit.to_csv(self.test_cases_1bit_xor_path, index=False, header=True)

        return df_test_cases_1bit

    def read_test_cases(self):
        df = pd.read_csv(self.test_cases_1bit_xor_path)

        return df

    def save_training_results(self, training_results):
        training_results_json = [{"Iteration": i+1, "Loss": loss} for i, loss in training_results]
        df_test_cases_1bit = pd.DataFrame(training_results_json)
        df_test_cases_1bit.to_csv(self.training_results_path, index=False, header=True)

    def plot_training_loss(self):
        csv_data = pd.read_csv(self.training_results_path)
        df = pd.DataFrame(csv_data, columns=["Iteration", "Loss"])
        plt.plot(df["Iteration"], df["Loss"])

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")

        plt.savefig(self.plot_training_loss_path)
        plt.show()

    def plot_testing_results(self, count_success, count_failure):
        labels = ['Success', 'Failed']
        x = [count_success, count_failure]
        colors = ['#00FF00', '#FF0000']

        _, ax1 = plt.subplots()

        ax1.pie(x=x, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')

        plt.title("Testing Results")
        plt.savefig(self.plot_testing_results_path)
        plt.show()


if __name__ == "__main__":
    data = Data()
    data.generate_test_dataset()
