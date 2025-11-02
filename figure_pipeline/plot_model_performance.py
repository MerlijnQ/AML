import json
import os
from typing import List, Optional
import matplotlib.pyplot as plt


class ModelPerformancePlotter:
    """
    Load model results from JSON files and plot RMSE,
    aleatoric uncertainty, and epistemic uncertainty across
    all time windows.
    """

    def __init__(self) -> None:
        """
        Initializes the ModelPerformancePlotter by loading model
        result files and preparing RMSE and uncertainty data.
        """
        self.directory: str = "model_results/"
        self.output_directory: str = "images/model_performance/"
        self.resultfiles: List[str] = [
            "results_size_24.json",
            "results_size_48.json",
            "results_size_72.json",
            "results_size_120.json"
        ]
        self.time_window_names: List[str] = [
            "24 hours",
            "48 hours",
            "72 hours",
            "120 hours"
        ]
        self.nr_of_features: int = 0
        self.RMSE: List[List[float]] = []
        self.aleatoric_uncertainty: List[List[float]] = []
        self.epistemic_uncertainty: List[List[float]] = []
        self.nr_features_in_run: List[int] = []

        self.__set_RMSE_files()

    def __set_RMSE_files(self) -> None:
        """
        Loads RMSE, aleatoric uncertainty, and epistemic uncertainty
        from JSON files and prepares the feature index for plotting.
        """
        for file_name in self.resultfiles:
            with open(os.path.join(self.directory, file_name), 'r') as f:
                window_results_json = json.load(f)

            self.RMSE.append(window_results_json["accuracy"][0])
            self.aleatoric_uncertainty.append(window_results_json["aleatoric_uncertainty"][0])
            self.epistemic_uncertainty.append(window_results_json["epistemic_uncertainty"][0])

        # Set feature run labels in descending order
        self.nr_of_features = len(self.RMSE[0])
        self.nr_features_in_run = [x for x in range(self.nr_of_features, 0, -1)]

    def __plot_and_save(
        self,
        y_values_list: List[List[float]],
        y_axis_label: str,
        output_file_name: str,
        show: bool = True
    ) -> None:
        """
        Creates a line plot for the provided y-values and saves it to a PDF.

        Args:
            y_values_list (List[List[float]]): List of lists containing values to plot per time window.
            y_axis_label (str): Label for the y-axis.
            output_file_name (str): Name of the output file (without extension).
            show (bool, optional): Whether to display the figure. Defaults to True.
        """
        plt.figure(figsize=(10, 5))

        # Plot each line
        for i, values in enumerate(y_values_list):
            plt.plot(self.nr_features_in_run, values, label=self.time_window_names[i])

        plt.xlabel("Number of features", fontsize=14)
        plt.ylabel(y_axis_label, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(1, self.nr_of_features)
        plt.gca().invert_xaxis()
        plt.legend(fontsize=12)

        os.makedirs(self.output_directory, exist_ok=True)
        plt.savefig(os.path.join(self.output_directory, output_file_name + ".pdf"))

        if show:
            plt.show()

    def plot_RMSE(self, show: bool = True) -> None:
        """
        Plots RMSE values for all time windows.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        self.__plot_and_save(
            self.RMSE,
            "RMSE [MWh]",
            "RMSE_plots",
            show=show
        )

    def plot_aleatoric_uncertainty(self, show: bool = True) -> None:
        """
        Plots aleatoric uncertainty for all time windows.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        self.__plot_and_save(
            self.aleatoric_uncertainty,
            "Aleatoric Uncertainty [MWh]",
            "aleatoric_uncertainty_plots",
            show=show
        )

    def plot_epistemic_uncertainty(self, show: bool = True) -> None:
        """
        Plots epistemic uncertainty for all time windows.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        self.__plot_and_save(
            self.epistemic_uncertainty,
            "Epistemic Uncertainty [MWh]",
            "epistemic_uncertainty_plots",
            show=show
        )


if __name__ == "__main__":
    modelPerformancePlotter = ModelPerformancePlotter()
    modelPerformancePlotter.plot_RMSE()
    modelPerformancePlotter.plot_aleatoric_uncertainty()
    modelPerformancePlotter.plot_epistemic_uncertainty()
