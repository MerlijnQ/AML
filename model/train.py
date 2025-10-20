from model.loss import customELBO, gnll
from model.model import DHBCNN
from torch.utils.data import DataLoader
import torch


class TrainTest():
    def __init__(self, max_epochs=30):
        self.max_epochs = max_epochs
        # self.criterion = customELBO(max_epochs)
        self.criterion = gnll() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self, model:DHBCNN, val_loader:DataLoader, epoch):
        model.eval()

        total_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                mu, sigma = model(X)
                if torch.isnan(mu).any():
                    print("NaN detected in mu!")
                    break
                if torch.isnan(sigma).any():
                    print("NaN detected in sigma!")
                    break
                loss = self.criterion(mu, sigma, y)
                if torch.isnan(loss):
                    print("Warning: NaN detected in validation loss")
                if torch.isinf(loss):
                    print("Warning: Inf detected in validation loss")
                total_loss += loss.item()
        #Note that the val and train loader need batches
        average_loss = total_loss / len(val_loader)
        print("Test loss: {}".format(average_loss))
        return average_loss

    def train(self, model:DHBCNN, train_loader:DataLoader, val_loader:DataLoader, lr=1e-4):
        model.to(self.device)
        best_model = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best = float("inf")
        no_improvement = 0

        for epoch in range(0, self.max_epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                mu, sigma = model(X)
                loss = self.criterion(mu, sigma, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Clip to prevent BOOM!
                optimizer.step()
            test_loss = self.test(model, val_loader, epoch)

            if test_loss < best:
                best = test_loss
                no_improvement = 0
                best_model = model
            else:
                no_improvement += 1

            if no_improvement == 5:
                break
        return best_model