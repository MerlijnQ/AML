import json
import random
import matplotlib.pyplot as plt
import os

# load the results data
with open('model_data.json', 'r') as f:
    results = json.load(f)

features = results["discarded"]
epistemic_uncertainty = results["epistemic_uncertainty"]
alaetoric_uncertainty = results["aleatoric_uncertainty"]
accuracy = results["accuracy"]

nr_features = len(features[0])
nr_window_sizes = 3

# define x-axis values from nr_features to 0
nr_features_values = [x for x in range(nr_features, 0, -1)]

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
        print(y_values)
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

# Call for accuracy plot
for i in range(3):
    plot_and_save(nr_features_values, [accuracy[i]], ["RMSE"], ['#1f77b4'], "RMSE_plot_" + str(i) + ".pdf")

# Call for Aleatoric and Epistemic plot
for i in range(nr_window_sizes):
    plot_and_save(
        nr_features_values,
        [alaetoric_uncertainty[i], epistemic_uncertainty[i]],
        ["Aleatoric Uncertainty", "Epistemic Uncertainty"],
        ['#ff7f0e', '#2ca02c'],
        "Uncertainty_plot_" + str(i) + ".pdf",
    )