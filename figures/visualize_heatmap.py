import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

with open('results_size_24.json', 'r') as f:
    results = json.load(f)

features = results["discarded"]
epistemic_uncertainty = results["epistemic_uncertainty"]
alaetoric_uncertainty = results["aleatoric_uncertainty"]
shap_values = results["shap_values"]

nr_features = len(shap_values[0])

def create_heatmap(shap_result, subtitle=""):
    # retrieve the data to form a heatmap. 
    # save the data structure in such a way that sns.heatmap knows what to do with it
    def get_heatmap_data(shap_result):
        global nr_features
        # global 
        heatmap_data = [[0, 0] for i in range(nr_features)]
        nr_features = len(shap_result)

        # add the rows to the data column
        # a row here is all values corresponding to all features to a certain run
        for run_nr, pairs in enumerate(shap_result):
            number_row = [pair[1] for pair in pairs]
            # insert a zero to the row in front until full
            for i in range(run_nr):
                number_row = [float('NaN')] + number_row

            heatmap_data[run_nr] = number_row

        # return the inverse, so each row has values of all runs of the same feature
        return [list(row) for row in zip(*heatmap_data)]

    def plot_heatmap():
        labels = [label[0] for label in shap_values[0][0]]
        # annot_labels = [[("" if val == 0 else val) for val in row] for row in heatmap_data]
        annot_labels = [
            ["" if val == float('NaN') else round(val, 4) for val in row]
            for row in heatmap_data
        ]
        print(annot_labels)
        # sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", yticklabels=labels)
        sns.heatmap(heatmap_data, annot=annot_labels, fmt="", cmap="coolwarm", cbar=False, yticklabels=labels)
        plt.xticks(ticks=[x + 0.5 for x in range(nr_features)], labels=list(range(nr_features, 0, -1)))
        plt.xlabel("nr_features")
        plt.ylabel("features")
        save_directory = 'heatmap'
        os.makedirs(save_directory, exist_ok=True)
        plt.savefig(save_directory + "/heatmap" + subtitle + ".pdf")
        plt.show()
    heatmap_data = get_heatmap_data(shap_result)
    plot_heatmap()

create_heatmap(results["shap_values"][0], "T0")
create_heatmap(results["shap_values"][1], "T1")
create_heatmap(results["shap_values"][2], "T2")