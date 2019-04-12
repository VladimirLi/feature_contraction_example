import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):
    for exp_name, accuracy in results.items():
        accuracy *= 100

        mean = np.mean(accuracy, axis=0)
        std = np.std(accuracy, axis=0)

        mean = 100-mean

        plt.errorbar(range(1,len(mean)+1), mean, yerr=std, label=exp_name)

    plt.ylabel("Test error [%]")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

