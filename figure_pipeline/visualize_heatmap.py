# open the data file
# determine the order of feature importance for the heatmap

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

class HeatMap:
    def __init__(self, data_file_name):
        """
        """
        self.data = self._load_data(data_file_name)
        self.feature_sets = self._get_feature_sets()
        self.labels = self._get_feature_importance_order()
        self.nr_features = len(self.labels)
        self.heatmap = self._create_heatmap_data()
        self.annot_labels = self._get_annot_labels()
        self.save_directory = 'images/heatmaps/'
        os.makedirs(self.save_directory, exist_ok=True)
        self.fontsize=20

        
    def _create_heatmap_data(self):
        # structure heatmap:
        # [[f1r1, f1r2, ..., f1rn], [f2...], [fnr1, fnr2, ..., fnrn]]
        nr_features = len(self.labels)
        heatmap = [[float('Nan') for j in range(nr_features)] for i in range(nr_features)]

        # for each run, save the corresponding feature values
        for run_nr in range(len(self.data)):
            run = self.data[run_nr]
            run_dict = dict(run)

            # for each feature in the feature importance label, save its values to the heatmap:
            for feature_idx, feature_name in enumerate(self.labels):
                if feature_name in run_dict:
                    # add the value to the heatmap
                    heatmap[feature_idx][run_nr] = run_dict[feature_name]

        return heatmap
    
    def _load_data(self, file_name):
        with open(file_name, 'r') as f:
            results = json.load(f)
        heatmap_data = results["shap_values"][0]

        # take hour_sin and hour_cos together. 
        for run_idx, run in enumerate(heatmap_data):
            new_run = []
            hour_sin = float('NaN')
            hour_cos = float('NaN')
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
    
    def _get_annot_labels(self):
        holiday_conversion_dict = {
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
        annot_labels = self.labels
        for label_idx, label in enumerate(self.labels):
            if label in holiday_conversion_dict:
                annot_labels[label_idx] = holiday_conversion_dict[label]
        return annot_labels
    
    def _get_feature_sets(self):
        return [set(f[0] for f in step) for step in self.data]

    def _get_feature_importance_order(self):
        """
        Determine the order in which features disappear.

        Returns:
        - List of feature names in the order they disappear.
        """
        if not self.feature_sets:
            return []

        disappearance_order = []
        remaining = self.feature_sets[0].copy()

        for step in self.feature_sets[1:]:
            disappeared = remaining - step
            disappearance_order.extend(disappeared)
            remaining = step

        # Any remaining features are the last to disappear
        disappearance_order.extend(remaining)
        return disappearance_order
    
    def plot_heatmap(self, subtitle):
        # create figure
        plt.figure(figsize=(16, 13))

        # create heatmap
        sns.heatmap(self.heatmap, annot=False, cmap="coolwarm", cbar=True, yticklabels=self.annot_labels)

        # figure config
        plt.xticks(ticks=[x + 0.5 for x in range(self.nr_features)], 
                   labels=list(range(self.nr_features, 0, -1)))
        plt.xlabel("Number of Features", fontsize=self.fontsize)
        plt.ylabel("Features", fontsize=self.fontsize)
        
        save_directory = 'heatmap'

        # save figure
        plt.savefig(self.save_directory + "heatmap_" + subtitle + ".pdf")
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