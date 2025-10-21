

# input: [window_24, window_48, window_72]: 3
#           where window = [feature_values_1, feature_values_2, ..]: nr_features
#           where feature_values = [(feature_name, shap_value), (), ..]: nr_features->1
nr_features = 20

def create_window():
    return []

def create_random_input():
    windows = [create_window for i in range()]

    return windows

randomized_input = create_random_input()