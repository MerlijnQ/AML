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
        
        self.image_output_dir = 'images/'

    def _load_results(self):
        """Load JSON results data from file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)

    def plot_and_save(self, x_values, y_values_list, labels, colors, filename, y_label, directory, show):
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
            plt.plot(x_values, y_values, label=label, color=color)
        plt.xlabel("Number of features")
        plt.ylabel(y_label)
        plt.xticks(x_values)
        plt.gca().invert_xaxis()
        plt.xlim(x_values[0], x_values[-1])
        # plt.legend(loc='lower left')
        
        plt.tight_layout()
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, filename))
        if show:
            plt.show()
        plt.close()

    def generate_plots(self, show=True):
        """Generate all required plots from loaded data."""
        for i in range(self.nr_window_sizes):
            # Accuracy plot
            self.plot_and_save(
                self.nr_features_values,
                [self.accuracy[i]],
                ["RMSE"],
                ['#1f77b4'],
                # f"RMSE_plot_{i}.pdf"
                f"RMSE_plot_{self.window_plot_name}.pdf",
                "RMSE [MWh]",
                self.image_output_dir + "accuracy_plots",
                show
            )

            # Aleatoric plot
            self.plot_and_save(
                self.nr_features_values,
                [self.alaetoric_uncertainty[i]],
                ["Aleatoric Uncertainty"],
                ['#2ca02c'],
                # f"Uncertainty_plot_aleatoric_{i}.pdf"
                f"Uncertainty_plot_aleatoric_{self.window_plot_name}.pdf",
                "Alaetoric uncertainty [MWh]",
                self.image_output_dir + "uncertainty_plots",
                show
            )

            # Epistemic plot
            self.plot_and_save(
                self.nr_features_values,
                [self.epistemic_uncertainty[i]],
                ["Epistemic Uncertainty"],
                ['#ff7f0e'],
                # f"Uncertainty_plot_epistemic_{i}.pdf"
                f"Uncertainty_plot_epistemic_{self.window_plot_name}.pdf",
                "Epistemic uncertainty [MWh]",
                self.image_output_dir + "uncertainty_plots",
                show
            )


if __name__ == "__main__":
    # Window size 24
    plotter = UncertaintyPlotter(results_file="model_results/results_size_24.json", window_plot_name="window_24")
    plotter.generate_plots(show=False)

    # Window size 48
    plotter = UncertaintyPlotter(results_file="model_results/results_size_48.json", window_plot_name="window_48")
    plotter.generate_plots(show=False)

    # Window size 72
    plotter = UncertaintyPlotter(results_file="model_results/results_size_72.json", window_plot_name="window_72")
    plotter.generate_plots(show=False)

    # Window size 120
    plotter = UncertaintyPlotter(results_file="model_results/results_size_120.json", window_plot_name="window_120")
    plotter.generate_plots(show=False)