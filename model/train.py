from model.loss import gnll
from model.model import DHBCNN
from torch.utils.data import DataLoader
import torch
import copy
import torch.nn as nn


class TrainTest():
    def __init__(self, max_epochs:int=100, warm_up:int=10)->None:
        
        """Initialize criterions for training.

        Args:
            max_epochs (int, optional): Maximum number of epochs during training. Defaults to 100.
            warm_up (int, optional): Warm-up phase durations expressed in epochs. Defaults to 10.
        """

        self.max_epochs = max_epochs
        # self.criterion = customELBO(max_epochs)
        self.criterion = gnll() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.warm_up = warm_up

    def test(self, model:DHBCNN, val_loader:DataLoader, epoch:int)->float:
        
        """Test the model performance using the criterion and the validation data to asses
            early stopping possibilities.

        Args:
            model (DHBCNN): A dynamically scaled two headed pytorch neural network.
            val_loader (DataLoader): A dataloader containing the validation data.
            epoch (int): The current training step.

        Returns:
            float: Validation loss.
        """

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
                loss = self.criterion(mu, sigma, y, epoch)
                if torch.isnan(loss):
                    print("Warning: NaN detected in validation loss")
                if torch.isinf(loss):
                    print("Warning: Inf detected in validation loss")
                total_loss += loss.item()
        #Note that the val and train loader need batches
        average_loss = total_loss / len(val_loader)
        print("Test loss: {}".format(average_loss))
        return average_loss

    def train(self, model:DHBCNN, train_loader:DataLoader, val_loader:DataLoader, lr:float=1e-4)->nn.Module:
        
        """A training loop including early stopping and warm-up phase for a two headed model with a prediction and variance head.

        Args:
            model (DHBCNN): A dynamically scaled two headed pytorch neural network.
            train_loader (DataLoader): A dataloader containing the training data.
            val_loader (DataLoader): A dataloader object containing the validation data.
            lr (float, optional): The learning rate of training process. This will be optimized using CosineAnnealing.
            Defaults to 1e-4.

        Returns:
            nn.Module: A trained dynamically scaled two headed pytorch neural network.
        """

        model.to(self.device)
        best_model = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs) #This makes sure the lr goes down with time

        best = float("inf")
        no_improvement = 0

        for epoch in range(0, self.max_epochs):
            model.train()
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                mu, sigma = model(X)
                loss = self.criterion(mu, sigma, y, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Clip to prevent BOOM!
                optimizer.step()

            scheduler.step()
            test_loss = self.test(model, val_loader, epoch)

            if not epoch < self.warm_up: #Only start early stopping after warm up and we only train on gnll
                if test_loss < best:
                    best = test_loss
                    no_improvement = 0
                    best_model = copy.deepcopy(model.state_dict())
                else:
                    no_improvement += 1

                if no_improvement == 5:
                    print("Early stopping triggered")
                    break

        if best_model is not None:
            model.load_state_dict(best_model)

        return model