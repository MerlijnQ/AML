import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from typing import List, Dict, Set, Any, Union


class HeatMap:
    """
    Class for creating and plotting heatmaps of feature importance
    based on SHAP values
    """

    def __init__(self, data_file_name: str) -> None:
        """
        Initializes the HeatMap object by loading data and preparing
        heatmap structure, feature ordering, and annotation labels.

        Args:
            data_file_name (str): Path to the JSON file containing SHAP values.
        """
        self.data: List[List[List[Union[str, float]]]] = self._load_data(data_file_name)
        self.feature_sets: List[Set[str]] = self._get_feature_sets()
        self.labels: List[str] = self._get_feature_importance_order()
        self.nr_features: int = len(self.labels)
        self.heatmap: List[List[float]] = self._create_heatmap_data()
        self.annot_labels: List[str] = self._get_annot_labels()
        self.save_directory: str = 'images/heatmaps/'
        os.makedirs(self.save_directory, exist_ok=True)
        self.fontsize: int = 20

    def _load_data(self, file_name: str) -> List[List[List[Union[str, float]]]]:
        """
        Loads SHAP values from a JSON file, combining hour_sin and hour_cos
        into a single feature 'hour_avg'.

        Args:
            file_name (str): Path to the JSON file.

        Returns:
            List[List[List[Union[str, float]]]]: List of runs containing feature-value pairs.
        """
        with open(file_name, 'r') as f:
            results: Dict[str, Any] = json.load(f)
        heatmap_data: List[List[List[Union[str, float]]]] = results["shap_values"][0]

        # Combine hour_sin and hour_cos into hour_avg
        for run_idx, run in enumerate(heatmap_data):
            new_run: List[List[Union[str, float]]] = []
            hour_sin: float = float('NaN')
            hour_cos: float = float('NaN')

            for feature_idx, feature in enumerate(run):
                if feature[0] == "hour_sin":
                    hour_sin = feature[1]
                elif feature[0] == "hour_cos":
                    hour_cos = feature[1]
                else:
                    new_run.append(feature)

            mean = np.nanmean([hour_sin, hour_cos])
            if not np.isnan(mean):
                new_run.append(["hour_avg", mean])

            heatmap_data[run_idx] = new_run

        return heatmap_data

    def _get_feature_sets(self) -> List[Set[str]]:
        """
        Returns a list of feature sets for each run.

        Returns:
            List[Set[str]]: List of sets of feature names per run.
        """
        return [set(f[0] for f in step) for step in self.data]

    def _get_feature_importance_order(self) -> List[str]:
        """
        Determines the order in which features disappear across runs.

        Returns:
            List[str]: Feature names in order of disappearance.
        """
        if not self.feature_sets:
            return []

        disappearance_order: List[str] = []
        remaining: Set[str] = self.feature_sets[0].copy()

        for step in self.feature_sets[1:]:
            disappeared: Set[str] = remaining - step
            disappearance_order.extend(disappeared)
            remaining = step

        disappearance_order.extend(remaining)
        return disappearance_order

    def _create_heatmap_data(self) -> List[List[float]]:
        """
        Structures heatmap data as a 2D list for plotting.
        Rows correspond to features (according to disappearance order),
        columns correspond to runs.

        Returns:
            List[List[float]]: 2D list with feature values for each run.
        """
        nr_features: int = len(self.labels)
        heatmap: List[List[float]] = [[float('NaN') for _ in range(nr_features)] for _ in range(nr_features)]

        for run_nr, run in enumerate(self.data):
            run_dict: Dict[str, float] = dict(run)
            for feature_idx, feature_name in enumerate(self.labels):
                if feature_name in run_dict:
                    heatmap[feature_idx][run_nr] = run_dict[feature_name]

        return heatmap

    def _get_annot_labels(self) -> List[str]:
        """
        Generates annotation labels for the heatmap, converting
        holiday IDs into readable names where applicable.

        Returns:
            List[str]: List of labels for heatmap y-axis.
        """
        holiday_conversion_dict: Dict[str, str] = {
            "Holiday_ID1": "New Year",
            "Holiday_ID2": "Martyrs' Day",
            "Holiday_ID3": "Carnival Saturday",
            "Holiday_ID4": "Carnival Sunday",
            "Holiday_ID5": "Carnival Monday",
            "Holiday_ID6": "Carnival Tuesday",
            "Holiday_ID7": "Ash Wednesday",
            "Holiday_ID8": "Holy Thursday",
            "Holiday_ID9": "Good Friday",
            "Holiday_ID10": "Holy Saturday",
            "Holiday_ID11": "Resurrection Sunday",
            "Holiday_ID12": "Labor Day",
            "Holiday_ID13": "Foundation of Old P.",
            "Holiday_ID14": "Separation of Colom.",
            "Holiday_ID15": "Flag Day",
            "Holiday_ID16": "Patriotic Comm.",
            "Holiday_ID17": "First Cry of Indepen.",
            "Holiday_ID18": "P. indepen. of Spain",
            "Holiday_ID19": "Mother's Day",
            "Holiday_ID20": "Christmas Eve",
            "Holiday_ID21": "Christmas",
            "Holiday_ID22": "New Year's Eve"
        }

        annot_labels: List[str] = self.labels.copy()
        for idx, label in enumerate(self.labels):
            if label in holiday_conversion_dict:
                annot_labels[idx] = holiday_conversion_dict[label]

        return annot_labels

    def plot_heatmap(self, subtitle: str) -> None:
        """
        Plots the heatmap of feature importance values.

        Args:
            subtitle (str): Substring to include in saved file name.
        """
        plt.figure(figsize=(16, 13))
        sns.heatmap(self.heatmap, annot=False, cmap="coolwarm", cbar=True, yticklabels=self.annot_labels)

        plt.xticks(
            ticks=[x + 0.5 for x in range(self.nr_features)],
            labels=list(range(self.nr_features, 0, -1))
        )
        plt.xlabel("Number of Features", fontsize=self.fontsize)
        plt.ylabel("Features", fontsize=self.fontsize)

        plt.savefig(os.path.join(self.save_directory, f"heatmap_{subtitle}.pdf"))
        # plt.show()


if __name__ == "__main__":
    heatmap_24 = HeatMap('model_results/results_size_24.json')
    heatmap_24.plot_heatmap("window_24")

    heatmap_48 = HeatMap('model_results/results_size_48.json')
    heatmap_48.plot_heatmap("window_48")

    heatmap_72 = HeatMap('model_results/results_size_72.json')
    heatmap_72.plot_heatmap("window_72")

    heatmap_120 = HeatMap('model_results/results_size_120.json')
    heatmap_120.plot_heatmap("window_120")
