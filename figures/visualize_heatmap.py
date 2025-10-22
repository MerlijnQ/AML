import json
import matplotlib.pyplot as plt
import seaborn as sns

with open('model_data.json', 'r') as f:
    results = json.load(f)

features = results["discarded"]
epistemic_uncertainty = results["epistemic_uncertainty"]
alaetoric_uncertainty = results["aleatoric_uncertainty"]
shap_values = results["shap_values"]

nr_features = len(shap_values[0])


data = [
    [["nat_demand", 0.002346440078836167],
     ["T2M_toc", 0.013409032346535243]],
    [["T2M_toc", 0.039039051532745374],
     [None, None]]
]

# change data structure in such a way that sns.heatmap knows what to do with it
shap_result = results["shap_values"][0]
nr_features = len(shap_result)
shap_data = [[0, 0] for i in range(nr_features)]

for run_nr, pairs in enumerate(shap_result):
    # add a zero pair to the pair until full
    number_row = [kaas[1] for kaas in pairs]
    for i in range(run_nr):
        # pairs = [[['kaas', 0]] + pairs]
        number_row = [0] + number_row
    shap_data[run_nr] = number_row

# now inverse the matrix
shap_data = [list(row) for row in zip(*shap_data)]
print(shap_data)

labels = [label[0] for label in shap_result[0]]


sns.heatmap(shap_data, annot=True, cmap="coolwarm", yticklabels=labels)
plt.xlabel("nr_features")
plt.ylabel("features")
plt.show()
# input: [window_24, window_48, window_72]: 3
#           where window = [feature_values_1, feature_values_2, ..]: nr_features
#           where feature_values = [(feature_name, shap_value), (), ..]: nr_features->1
# nr_features = 20

# def create_window():
#     return []

# def create_random_input():
#     windows = [create_window for i in range()]

#     return windows

# randomized_input = create_random_input()