import torch.nn as nn
import torch
from model.model import DHBCNN
from model.train import TrainTest

class ensemble_DHBCNN(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        for model in models:
            model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return prediction

    @torch.no_grad
    def evaluate(self, test_loader):
        mus = []
        sigmas = []
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

        epistemic = epistemic.mean().item()
        aleatoric = aleatoric.mean().item()
        rmse = torch.sqrt(torch.mean((prediction - y) ** 2)).item()

        return rmse, aleatoric, epistemic


class create_ensemble():
    def __init__(self, n_features, window_size, train_loader, val_loader):
        super().__init__()
        seeds = [5, 8, 64, 32, 5]
        self.models = []
        trainer = TrainTest(5)
        for seed in seeds:
            torch.manual_seed(seed)
            new_model = DHBCNN(n_features, window_size)
            trained_model = trainer.train(new_model, train_loader, val_loader)
            self.models.append(trained_model)

    def get_ensemble_model(self):
        return ensemble_DHBCNN(self.models)