import random
import matplotlib.pyplot as plt

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

# Extend y-values to match x-values by adding a zero at the start
RMSE_plot = RMSE
aleatoric_plot = aleatoric
epistemic_plot = epistemic

plt.figure(figsize=(10, 5))
plt.plot(x_values, RMSE_plot, label="RMSE", marker='o')
plt.plot(x_values, aleatoric_plot, label="Aleatoric Uncertainty", marker='s')
plt.plot(x_values, epistemic_plot, label="Epistemic Uncertainty", marker='^')
plt.title("Randomly Generated Uncertainties")
plt.xlabel("nr_features")
plt.ylabel("Value")
plt.xticks(x_values)
plt.gca().invert_xaxis()  # Reverse x-axis
plt.legend()
plt.grid(True)
plt.show()
