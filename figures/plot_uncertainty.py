import json
import os
import matplotlib.pyplot as plt


class UncertaintyPlotter:
    def __init__(self, results_file='results_size_24.json', window_plot_name=""):
        self.results_file = results_file
        self.results = self._load_results()
        self.features = self.results["discarded"]
        self.epistemic_uncertainty = self.results["epistemic_uncertainty"]
        self.alaetoric_uncertainty = self.results["aleatoric_uncertainty"]
        self.accuracy = self.results["accuracy"]
        self.window_plot_name = window_plot_name

        self.nr_features = len(self.features[0])
        self.nr_window_sizes = 1
        self.nr_features_values = [x for x in range(self.nr_features, 0, -1)]

    def _load_results(self):
        """Load JSON results data from file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)

    def plot_and_save(self, x_values, y_values_list, labels, colors, filename, directory="uncertainty_plots"):
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
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, filename))
        plt.show()
        plt.close()

    def generate_plots(self):
        """Generate all required plots from loaded data."""
        for i in range(self.nr_window_sizes):
            # Accuracy plot
            self.plot_and_save(
                self.nr_features_values,
                [self.accuracy[i]],
                ["RMSE"],
                ['#1f77b4'],
                # f"RMSE_plot_{i}.pdf"
                f"RMSE_plot_{self.window_plot_name}.pdf"
            )

            # Aleatoric plot
            self.plot_and_save(
                self.nr_features_values,
                [self.alaetoric_uncertainty[i]],
                ["Aleatoric Uncertainty"],
                ['#2ca02c'],
                # f"Uncertainty_plot_aleatoric_{i}.pdf"
                f"Uncertainty_plot_aleatoric_{self.window_plot_name}.pdf"
            )

            # Epistemic plot
            self.plot_and_save(
                self.nr_features_values,
                [self.epistemic_uncertainty[i]],
                ["Epistemic Uncertainty"],
                ['#ff7f0e'],
                # f"Uncertainty_plot_epistemic_{i}.pdf"
                f"Uncertainty_plot_epistemic_{self.window_plot_name}.pdf"
            )


if __name__ == "__main__":
    # Window size 24
    plotter = UncertaintyPlotter(results_file="results_size_24.json", window_plot_name="window_24")
    plotter.generate_plots()

    # Window size 48
    plotter = UncertaintyPlotter(results_file="results_size_48.json", window_plot_name="window_48")
    plotter.generate_plots()

    # Window size 72
    plotter = UncertaintyPlotter(results_file="results_size_72.json", window_plot_name="window_72")
    plotter.generate_plots()

    # Window size 120
    plotter = UncertaintyPlotter(results_file="results_size_120.json", window_plot_name="window_120")
    plotter.generate_plots()


# # Call for Aleatoric and Epistemic plot
# for i in range(nr_window_sizes):
#     plot_and_save(
#         nr_features_values,
#         [alaetoric_uncertainty[i], epistemic_uncertainty[i]],
#         ["Aleatoric Uncertainty", "Epistemic Uncertainty"],
#         ['#ff7f0e', '#2ca02c'],
#         "Uncertainty_plot_" + str(i) + ".pdf",
#     )