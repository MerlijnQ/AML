from dataloader.dataloader import DataLoaderTimeSeries
from model.ensemble import create_ensemble
from shap_explainer import SHAPExplainer
import json
import sys

if __name__ == "__main__":
    model_data = {
        "epistemic_uncertainty": [[]],
        "aleatoric_uncertainty": [[]],
        "accuracy": [[]],
        "discarded": [[]],
        "shap_values": [[]]
    }

    args = sys.argv[1:]
    window_sizes = [int(args[0])]

    explainer = SHAPExplainer()

    for s in range(len(window_sizes)):
        print(f"Window size: {window_sizes[s]}")

        data_loader = DataLoaderTimeSeries(window_sizes[s])

        n = len(data_loader.features)

        while n > 0:
            print(f"Number of features: {n}")

            # Create model
            ensemble_builder = create_ensemble(
                n,
                window_sizes[s],
                data_loader.train_loader,
                data_loader.validation_loader,
            )
            model = ensemble_builder.get_ensemble_model() # Model returns rmse, epistemic, aleatoric on prediction

            # Evaluate model
            rmse, epi, alea = model.evaluate(data_loader) # pass entire object to rescale for rmse calculation
            print(f"RMSE: {rmse}, epistemic: {epi}, aleatoric: {alea}")

            # Store uncertainties and accuracy
            model_data["epistemic_uncertainty"][s].append(epi)
            model_data["aleatoric_uncertainty"][s].append(alea)
            model_data["accuracy"][s].append(rmse)

            # Feature selection with SHAP
            current_shap_values, discarded_feature = explainer.explain_predictions(
                X_train=data_loader.train_loader,
                X_test=data_loader.test_loader,
                model=model,
                features=data_loader.features,
                apple_silicon=False
            )

            # Store shap values
            model_data["shap_values"][s].append(current_shap_values)

            # Store discarded feature
            model_data["discarded"][s].append(discarded_feature)

            # Remove the feature from the dataloader
            print(f"Discarding feature: {discarded_feature}")
            print("data: {}".format(model_data))

            if discarded_feature == "hour":
                data_loader.remove_feature("hour_sin")
                data_loader.remove_feature("hour_cos")
                n-=2
            else:
                data_loader.remove_feature(discarded_feature)
                n-=1
    
    with open(f"results_size_{window_sizes[s]}.json", "w") as f:
        json.dump(model_data, f, indent=4)

    # figures = make figure(uncertainty, Accuracy, Discarded)