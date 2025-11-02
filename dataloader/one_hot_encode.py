import pandas as pd

def one_hot_encode(feature: list[str], feature_data:pd.Series) -> pd.DataFrame:
    """
    Function that one hot encodes the data of a feature.

    Args:
        feature (str): the feature that needs to be one hot encoded.
        feature_data (pd.Series): the data of the feature.

    Returns:
        pd.DataFrame: the one hot encoded data of size (data length*unique values)
    """
    unique_values = list(feature_data.unique())
    unique_values.remove(0)
    unique_values.sort()

    column_names = [feature + str(i) for i in unique_values]
    data = dict()
    for i in column_names:
        data[i] = [] 
    for i in range(len(unique_values)):
        for j in range(len(feature_data)):
            if feature_data[j] == i+1:
                data[column_names[i]].append(1)
            else:
                data[column_names[i]].append(0)
    return pd.DataFrame(data)