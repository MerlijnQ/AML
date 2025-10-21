from dataloader import DataLoaderTimeSeries
from model.ensemble import create_ensemble
from shap_explainer import explain_predictions_test

if __name__ == "__main__":
    epistemic_uncertainty = [[],[],[]]
    aleatoric_uncertainty = [[],[],[]]
    accuracy = [[],[],[]]
    discarded = [[],[],[]]

    window_sizes = [24, 48, 72]

    for s in range(len(window_sizes)):
        print(f"Window size: {window_sizes[s]}")

        data_loader = DataLoaderTimeSeries(window_sizes[s])

        num_features = len(data_loader.features)

        # TODO REMOVE
        for n in range(0, num_features-2):
            data_loader.remove_feature(data_loader.get_feature_at_index(0))
        num_features = len(data_loader.features)

        for n in range(num_features, 0, -1):
            print(f"Number of features: {n}")

            # Create model
            ensemble_builder = create_ensemble(
                n,
                window_sizes[s],
                data_loader.train_loader,
                data_loader.validation_loader
            )
            model = ensemble_builder.get_ensemble_model() # Model returns rmse, epistemic, aleatoric on prediction

            # Evaluate model
            rmse, epi, alea = model.evaluate(data_loader.test_loader)
            print(f"RMSE: {rmse}, epistemic: {epi}, aleatoric: {alea}")

            # Store uncertainties and accuracy
            epistemic_uncertainty[s].append(epi)
            aleatoric_uncertainty[s].append(alea)
            accuracy[s].append(rmse)

            # Feature selection with SHAP
            # TODO REMOVE TEST
            discarded_feature = explain_predictions_test(
                X_train=data_loader.train_loader,
                X_test=data_loader.test_loader,
                model=model,
                features=data_loader.features)

            # Store discarded feature
            discarded[s].append(discarded_feature)

            # Remove the feature from the dataloader
            print(f"Discarding feature: {discarded_feature}")
            data_loader.remove_feature(discarded_feature)

    # figures = make figure(uncertainty, Accuracy, Discarded)
