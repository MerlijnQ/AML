import torch.nn as nn
import torch
from model.model import DHBCNN
from model.train import TrainTest
import numpy as np
import random

class ensemble_DHBCNN(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = nn.ModuleList(models)
        for model in models:
            model.eval()

    def forward(self, X):
        mus = []
        sigmas = []
        for model in self.models:   
            mu, sigma = model(X)
            mus.append(mu)
            sigmas.append(sigma)

        mus = torch.stack(mus, dim=0)
        sigmas = torch.stack(sigmas, dim=0)

        prediction = mus.mean(dim=0)

        # if prediction.dim() > 1 and prediction.shape[1] == 1:
        #     prediction = prediction.squeeze(1)

        return prediction

    @torch.no_grad
    def evaluate(self, data_loader):
        mus = []
        sigmas = []
        test_loader = data_loader.test_loader

        y = torch.cat([y.to(self.device) for _, y in test_loader], dim=0)
       
        for model in self.models:
            mu_model = []
            sigma_model = []
            for X, _ in test_loader:
                X = X.to(self.device)
                mu, sigma = model(X)
                mu_model.append(mu)
                sigma_model.append(sigma)
            mu_model =  torch.cat(mu_model, dim=0)
            sigma_model = torch.cat(sigma_model, dim=0)
            mus.append(mu_model)
            sigmas.append(sigma_model)
        mus = torch.stack(mus, dim=0)
        sigmas = torch.stack(sigmas, dim=0)


        prediction = mus.mean(dim=0)
        epistemic = mus.var(dim=0) 
        aleatoric = (sigmas**2).mean(dim=0)
        
        # epistemic = epistemic.mean().item()
        # aleatoric = aleatoric.mean().item()

        prediction = data_loader._inverse_transform_target(prediction) #Bring predictions back to original scale
        y = data_loader._inverse_transform_target(y)
        rmse = torch.sqrt(torch.mean((prediction - y) ** 2)).item()

        # Unscale uncertainties: convert variance → std to report in original units
        #The offset b (the mean shift) does not affect variance or standard deviation after linear transformation Z = aX + b as Var(Z) = a^2Var(X). Here a = W which scales our variance.
        aleatoric_std_unscaled = torch.sqrt(aleatoric) * data_loader.target_std 
        epistemic_std_unscaled = torch.sqrt(epistemic) * data_loader.target_std

        # Mean over all samples
        aleatoric_std_unscaled = aleatoric_std_unscaled.mean().item()
        epistemic_std_unscaled = epistemic_std_unscaled.mean().item()

        return rmse, aleatoric_std_unscaled, epistemic_std_unscaled


class create_ensemble():
    def __init__(self, n_features, window_size, train_loader, val_loader):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seeds = [5, 8, 64, 32, 42, 83, 27]
        self.models = []
        trainer = TrainTest()
        for seed in seeds:
            print("Training model with seed: {}".format(seed))
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
            new_model = DHBCNN(n_features, window_size).to(self.device) #Make sure to intialize a new model on the right device for each M
            trained_model = trainer.train(new_model, train_loader, val_loader)
            self.models.append(trained_model)

    def get_ensemble_model(self):
        return ensemble_DHBCNN(self.models)