import json
import os
import matplotlib.pyplot as plt


class ModelPerformancePlotter:
    def __init__(self):
        self.directory = "model_results/"
        self.output_directory = "images/model_performance/"
        self.resultfiles = [
            "results_size_24.json",
            "results_size_48.json",
            "results_size_72.json",
            "results_size_120.json"
        ]
        self.time_window_names = [
            "24 hours",
            "48 hours",
            "72 hours",
            "120 hours"
        ]
        self.nr_of_features = 0
        self.RMSE = []
        self.aleatoric_uncertainty = []
        self.epistemic_uncertainty = []
        self.nr_features_in_run = []
        self.__set_RMSE_files()
    
    def __set_RMSE_files(self):
        for file_name in self.resultfiles:
            # read files
            with open(self.directory + file_name, 'r') as f:
                window_results_json = json.load(f)
            self.RMSE.append(window_results_json["accuracy"][0])
            self.aleatoric_uncertainty.append(window_results_json["aleatoric_uncertainty"][0])
            self.epistemic_uncertainty.append(window_results_json["epistemic_uncertainty"][0])

        # set feature run labels
        self.nr_of_features = len(self.RMSE[0])
        self.nr_features_in_run = [x for x in range(self.nr_of_features, 0, -1)]

    def __plot_and_save(self, y_values_list, y_axis_label, output_file_name, show=True):
        plt.figure(figsize=(10, 5))

        # Plot each RMSE list
        for i, rmse_list in enumerate(y_values_list):
            plt.plot(self.nr_features_in_run, rmse_list, label=self.time_window_names[i])

        plt.xlabel("Number of features", fontsize=14)
        plt.ylabel(y_axis_label, fontsize=14)

        # Make tick labels larger too
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(1, self.nr_of_features)

        plt.gca().invert_xaxis()
        plt.legend(fontsize=12)

        os.makedirs(self.output_directory, exist_ok=True)
        plt.savefig(os.path.join(self.output_directory, output_file_name + ".pdf"))

        if show:
            plt.show()

    def plot_RMSE(self, show=True):
        self.__plot_and_save(
            self.RMSE,
            "RMSE [MWh]",
            "RMSE_plots",
            show=show
        )

    def plot_aleatoric_uncertainty(self, show=True):
        self.__plot_and_save(
            self.aleatoric_uncertainty,
            "Aleatoric Uncertainty [MWh]",
            "aleatoric_uncertainty_plots",
            show=show
        )

    def plot_epistemic_uncertainty(self, show=True):
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