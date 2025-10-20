from dataloader import DataLoaderTimeSeries
# from model.model import DHBCNN
# from model.train import TrainTest
from model.ensemble import create_ensemble
from shap_explainer import explain_predictions

if __name__ == "__main__":
    uncertainty = [[],[],[]]
    accuracy = [[],[],[]]
    discarded = [[],[],[]]

    window_sizes = [24, 48, 72]

    for s in range(len(window_sizes)):
        data_loader = DataLoaderTimeSeries(window_sizes[s])
        for n in range(len(data_loader.features), 0, -1):
            create_new_ensemble = create_ensemble(
                n, window_sizes[s], data_loader.train_loader, data_loader.validation_loader)
            model = create_new_ensemble.get_ensemble_model() #Model returns rmse, epistemic, aleatoric on prediction
            rmse, epi, alea = model.evaluate(data_loader.test_loader)
            # model = DHBCNN(n, window_sizes[s])
            # model = TrainTest().train(model, data_loader.train_loader, data_loader.validation_loader)
            # uncertainty[s].append(   predict uncertainty(model, DataLoader)   )
            # accuracy[s].append(      predict accuracy(model, DataLoader)      )
            discarded[s].append(explain_predictions(data_loader.train_loader, data_loader.validation_loader, model))
            discarded_feature = data_loader.get_feature_at_index(discarded[s][-1])
            data_loader.remove_feature(discarded_feature)

    # figures = make figure(uncertainty, Accuracy, Discarded)
