import shap
import torch
import numpy as np

# Wrapper for the predictions because SHAP requires the model to be deterministic
def shap_predict(model, X):
    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        mu, _ = model(X)
    return mu

# Explain the predictions and return the least important feature
def explain_predictions(X_train, X_test, model):
    background_distribution = torch.tensor(X_train[:100], dtype=torch.float32)
    test_tensor = torch.tensor(X_test[:10], dtype=torch.float32)

    shap_explainer = shap.GradientExplainer(model=lambda x: shap_predict(model, x), data=background_distribution)
    shap_values = shap_explainer.shap_values(test_tensor)

    global_importance = np.abs(shap_values).mean(axis=2)  # (samples, features)

    X_flat = test_tensor.mean(dim=2).numpy()

    shap.summary_plot(global_importance, X_flat)

    idx_least_important_feature = np.argmin(global_importance)

    return idx_least_important_feature
