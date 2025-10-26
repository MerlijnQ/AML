# open the data file
# determine the order of feature importance for the heatmap

import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

class HeatMap:
    def __init__(self, data_file_name):
        """
        """
        self.data = self._load_data(data_file_name)
        self.feature_sets = self._get_feature_sets()
        self.labels = self._get_feature_importance_order()
        self.nr_features = len(self.labels)
        self.heatmap = self._create_heatmap_data()
        
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
        with open('results_size_24.json', 'r') as f:
            results = json.load(f)
        return results["shap_values"][0]
    
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
        sns.heatmap(self.heatmap, annot=False, cmap="coolwarm", cbar=True, yticklabels=self.labels)

        # figure config
        plt.xticks(ticks=[x + 0.5 for x in range(self.nr_features)], labels=list(range(self.nr_features, 0, -1)))
        plt.xlabel("nr_features")
        plt.ylabel("features")
        save_directory = 'heatmap'

        # save figure
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(save_directory + "/heatmap_" + subtitle + ".pdf")
        # plt.show()


if __name__ == "__main__":
    heatmap_24 = HeatMap('results_size_24.json')
    heatmap_24.plot_heatmap("window_24")

    heatmap_48 = HeatMap('results_size_48.json')
    heatmap_48.plot_heatmap("window_48")

    heatmap_72 = HeatMap('results_size_72.json')
    heatmap_72.plot_heatmap("window_72")

    heatmap_120 = HeatMap('results_size_120.json')
    heatmap_120.plot_heatmap("window_120")