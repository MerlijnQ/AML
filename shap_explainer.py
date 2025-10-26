import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Wrapper for the predictions because SHAP requires the model to be deterministic
def shap_predict(model, X, input_shape):
    device = next(model.parameters()).device
    batch_size = X.shape[0]
    n_features, seq_len = input_shape

    X = torch.from_numpy(X).float().to(device)
    X = X.view(batch_size, n_features, seq_len)

    model.eval()
    torch.manual_seed(0)

    with torch.no_grad():
        pred = model(X)

    return pred.detach().cpu().numpy()

def get_samples_from_loader(loader, n_samples):
    samples = []
    for batch_X, _ in loader:
        samples.append(batch_X)
        if sum(s.shape[0] for s in samples) >= n_samples:
            break
    return torch.cat(samples, dim=0)[:n_samples]

# Function that explain the predictions and returns an array with features and shap values and least important feature.
# SHAP values for "hour_cos" and "hour_sin" are computed separately, and then averaged in the new "hour" feature.
# SHAP values for "hour_cos" and "hour_sin" are returned separately, but the least important feature is renamed "hour".
#
# NOTE: Only set apple_silicon=True if testing SHAP on a device with Apple Silicon.
#       This fixes a running error, but it will return different SHAP values.
def explain_predictions(X_train, X_test, model, features, apple_silicon=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    background_distribution = get_samples_from_loader(X_train, 100).to(device=device, dtype=torch.float32)
    test_tensor = get_samples_from_loader(X_test, 10).to(device=device, dtype=torch.float32)

    _, n_features, seq_len = background_distribution.shape
    input_shape = (n_features, seq_len)

    background_np = background_distribution.view(background_distribution.size(0), -1).cpu().numpy()
    test_np = test_tensor.view(test_tensor.size(0), -1).cpu().numpy()

    device = next(model.parameters()).device
    
    if not apple_silicon:
        shap_explainer = shap.GradientExplainer(
            model=model,
            data=background_distribution.to(device=device, dtype=torch.float32)
        )
        shap_values = shap_explainer.shap_values(test_tensor.to(device=device, dtype=torch.float32))

    else:
        shap_explainer = shap.KernelExplainer(
            model=lambda x: shap_predict(model, x, input_shape),
            data=background_np
        )
        shap_values = shap_explainer.shap_values(test_np)

    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.detach().cpu().numpy()

    global_importance = np.abs(shap_values)
    global_importance = global_importance.reshape(global_importance.shape[0], n_features, seq_len)
    global_importance = global_importance.mean(axis=2) # average over timesteps
    test_features_only = test_np.reshape(test_np.shape[0], n_features, seq_len).mean(axis=2)

    shap_values_obj = shap._explanation.Explanation(
        values=global_importance,
        data=test_features_only,
        feature_names=features
    )

    # Combine SHAP values of hour into one
    if "hour_sin" in shap_values_obj.feature_names and "hour_cos" in shap_values_obj.feature_names:
        values_df = pd.DataFrame(shap_values_obj.values, columns=shap_values_obj.feature_names)
        values_df["hour"] = values_df[["hour_sin", "hour_cos"]].mean(axis=1)
        values_df = values_df.drop(columns=["hour_sin", "hour_cos"])

        data_df = pd.DataFrame(shap_values_obj.data, columns=shap_values_obj.feature_names)
        data_df["hour"] = data_df[["hour_sin", "hour_cos"]].mean(axis=1)
        data_df = data_df.drop(columns=["hour_sin", "hour_cos"])

        shap_values_combined = shap.Explanation(
            values=values_df.values,
            base_values=shap_values_obj.base_values,
            data=data_df.values,
            feature_names=values_df.columns.tolist()
        )

    # # Plot original features
    # shap.plots.bar(shap_values_obj, max_display=n_features, show=False)

    # title = f"SHAP Summary ({n_features} features × {seq_len} timesteps)"
    # plt.title(title)
    # filename = f"shap_{n_features}_features_{seq_len}_steps.png"
    # os.makedirs("SHAP_plots", exist_ok=True)
    # save_path = os.path.join("SHAP_plots", filename)
    # plt.savefig(save_path, bbox_inches="tight", dpi=300)
    # plt.close()

    # Calculate original SHAP values tuples
    avg_importance = global_importance.mean(axis=0)
    feature_importance_tuples = [(feat, float(val)) for feat, val in zip(features, avg_importance)]

    # Get the least important feature (can return hour if the average of hour_cos and hour_sin is the lowest)
    if "hour_sin" in shap_values_obj.feature_names and "hour_cos" in shap_values_obj.feature_names:
        combined_avg_importance = np.mean(np.abs(shap_values_combined.values), axis=0)
        combined_feature_names = shap_values_combined.feature_names
        idx_least_important_feature = np.argmin(combined_avg_importance)
        least_important_feature = combined_feature_names[idx_least_important_feature]
    else:
        idx_least_important_feature = np.argmin(avg_importance)
        least_important_feature = features[idx_least_important_feature]

    return feature_importance_tuples, least_important_feature
