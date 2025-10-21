import shap
import torch
import numpy as np

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
        pred = model(X)[0]

    return pred.detach().cpu().numpy()

def get_samples_from_loader(loader, n_samples):
    samples = []
    for batch_X, _ in loader:
        samples.append(batch_X)
        if sum(s.shape[0] for s in samples) >= n_samples:
            break
    return torch.cat(samples, dim=0)[:n_samples]

# Explain the predictions and return the least important feature
def explain_predictions(X_train, X_test, model, features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    background_distribution = get_samples_from_loader(X_train, 100).to(device=device, dtype=torch.float32)
    test_tensor = get_samples_from_loader(X_test, 10).to(device=device, dtype=torch.float32)

    _, n_features, seq_len = background_distribution.shape
    input_shape = (n_features, seq_len)

    background_np = background_distribution.view(background_distribution.size(0), -1).cpu().numpy()
    test_np = test_tensor.view(test_tensor.size(0), -1).cpu().numpy()

    shap_explainer = shap.GradientExplainer(
        model=lambda x: shap_predict(model, x, input_shape),
        data=background_np
    )
    shap_values = shap_explainer.shap_values(test_np)

    global_importance = np.abs(shap_values)
    global_importance = global_importance.reshape(global_importance.shape[0], n_features, seq_len)
    global_importance = global_importance.mean(axis=2) # average over timesteps
    test_features_only = test_np.reshape(test_np.shape[0], n_features, seq_len).mean(axis=2)

    shap.summary_plot(global_importance, test_features_only, feature_names=features)

    avg_importance = global_importance.mean(axis=0)
    idx_least_important_feature = np.argmin(avg_importance)
    least_important_feature = features[idx_least_important_feature]

    return least_important_feature

# Test function with small sample sizes that works on Apple Silicon
def explain_predictions_test(X_train, X_test, model, features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    background_distribution = get_samples_from_loader(X_train, 1).to(device=device, dtype=torch.float32)
    test_tensor = get_samples_from_loader(X_test, 1).to(device=device, dtype=torch.float32)

    _, n_features, seq_len = background_distribution.shape
    input_shape = (n_features, seq_len)

    background_np = background_distribution.view(background_distribution.size(0), -1).cpu().numpy()
    test_np = test_tensor.view(test_tensor.size(0), -1).cpu().numpy()

    shap_explainer = shap.KernelExplainer(
        model=lambda x: shap_predict(model, x, input_shape),
        data=background_np
    )
    shap_values = shap_explainer.shap_values(test_np)

    global_importance = np.abs(shap_values)
    global_importance = global_importance.reshape(global_importance.shape[0], n_features, seq_len)
    global_importance = global_importance.mean(axis=2) # average over timesteps
    test_features_only = test_np.reshape(test_np.shape[0], n_features, seq_len).mean(axis=2)

    shap.summary_plot(global_importance, test_features_only, feature_names=features)

    avg_importance = global_importance.mean(axis=0)
    idx_least_important_feature = np.argmin(avg_importance)
    least_important_feature = features[idx_least_important_feature]

    return least_important_feature