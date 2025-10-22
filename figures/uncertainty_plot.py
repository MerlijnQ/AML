import random
import matplotlib.pyplot as plt
import os

# NOTE: Please double check if graph is not displayed in reserved: e.g. nr_features = 19 gives y-values for nr_features = 1
# input RMSE, aleatoric unc., epistemic unc.
# output: graph

def create_random_values(nr_features):
    return [random.random() for i in range(nr_features)]

nr_features = 20
RMSE = create_random_values(nr_features)
aleatoric = create_random_values(nr_features)
epistemic = create_random_values(nr_features)

# x-axis values from 0 to nr_features
x_values = list(range(nr_features))
x_values = [i + 1 for i in x_values]

# Extend y-values to match x-values by adding a zero at the start
RMSE_plot = RMSE
aleatoric_plot = aleatoric
epistemic_plot = epistemic

def plot_and_save(x_values, y_values_list, labels, colors, filename, directory="uncertainty_plots"):
    """
    Create a plot from given y-values, labels, and colors, then save it.

    Parameters:
    - x_values: list of x-axis values
    - y_values_list: list of lists of y-axis values (one list per line)
    - labels: list of labels for each line
    - colors: list of colors for each line
    - filename: name of the file to save the plot
    """
    plt.figure(figsize=(10, 5))
    for y_values, label, color in zip(y_values_list, labels, colors):
        plt.plot(x_values, y_values, label=label, color=color, marker='o')
    plt.xlabel("Number of features")
    plt.ylabel("Value")
    plt.xticks(x_values)
    plt.gca().invert_xaxis()
    plt.xlim(x_values[0], x_values[-1])
    plt.legend(loc='lower left')
    plt.tight_layout()
    if directory != "":
        os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + "/" + filename)
    plt.show()
    plt.close()

# Call for RMSE plot
plot_and_save(x_values, [RMSE_plot], ["RMSE"], ['#1f77b4'], "RMSE_plot.pdf")

# Call for Aleatoric and Epistemic plot
plot_and_save(
    x_values,
    [aleatoric_plot, epistemic_plot],
    ["Aleatoric Uncertainty", "Epistemic Uncertainty"],
    ['#ff7f0e', '#2ca02c'],
    "Uncertainty_plot.pdf",
)