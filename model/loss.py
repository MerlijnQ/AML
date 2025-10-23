import torch
import torch.nn as nn

class customELBO(nn.Module):
    def __init__(self, total_epochs=50):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.priors = {"mu_head": torch.tensor(1.0, device=self.device, dtype=torch.float32) ,
                       "sigma_head": torch.tensor(0.5, device=self.device, dtype=torch.float32)} #Only represent STD as we say mu=0 for both
        self.total_epochs = total_epochs
        self.eps = 1e-6
        
    def GNLL(self, mu, sigma, y):
        # Ensure sigma is valid
        sigma = torch.log1p(torch.exp(sigma))  # softplus for positivity
        sigma = torch.clamp(sigma, min=self.eps, max=1e3)  # avoid 0 and inf

        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            #We get [N, C] but as C is 1 we can lose it

        total = 0.5 * torch.log(2 * torch.pi * sigma**2) + ((mu - y)**2 / (2 * sigma**2 + self.eps))
        return total.mean()

    def KL(self, mu, sigma, sigma_p):
        # Ensure sigma is valid
        sigma = torch.log1p(torch.exp(sigma))  # softplus for positivity
        sigma = torch.clamp(sigma, min=self.eps, max=1e3)  # avoid 0 and inf
        # Ensure sigma_p is valid
        sigma_p = torch.log1p(torch.exp(sigma_p))  # softplus for positivity
        sigma_p = torch.clamp(sigma_p, min=self.eps, max=1e3)  # avoid 0 and inf
        # sigma = torch.clamp(sigma, min=-20, max=5) #Prevent NaN values with big sigma but still push it in the right direction.
        kl = torch.log(sigma_p/sigma + self.eps) + (sigma**2 + mu**2)/(2 * sigma_p**2 + self.eps) - 0.5
        kl = kl.sum()
        return kl 

    def warm_up_schedule(self, epoch, max_p):
        #Warmup in 30% of epochs (arbitrarely chosen)
        return min(max_p, epoch / (0.3 * self.total_epochs))  

    def get_kl(self, m, sigma_p, epoch, max_p):
        #Determine the strength of the KL penalty per head. 
        #To allow for warm up we want the model to explore first.
        #Therefore we increase the KL penalty with time per head.
        #Max penalty for sigma should be lower to prevent variance collapse in the model.
        penalty = self.warm_up_schedule(epoch, max_p)

        kl1 = self.KL(m.W_mu, m.W_sigma, sigma_p)
        kl2 = self.KL(m.b_mu, m.b_sigma, sigma_p)
        return (kl1 + kl2) * penalty
    
    def forward(self, mu, sigma, y, model, epoch, train=True):
        gnll = self.GNLL(mu, sigma, y)
        if train:
            kl1 = self.get_kl(model.mu, self.priors["mu_head"], epoch, max_p=1)
            kl2 = self.get_kl(model.sigma, self.priors["sigma_head"], epoch, max_p=0.1)
            total_kl = kl1 + kl2
            return gnll + total_kl
        return gnll
    
class gnll(nn.Module):
    def __init__(self, warm_up=10):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = 1e-6
        self.warm_up = warm_up

    def MSE(self, mu, y):
        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
        mse = torch.mean((mu - y) ** 2)
        return mse
        
    def GNLL(self, mu, sigma, y):
        # Ensure sigma is valid
        # sigma = torch.log1p(torch.exp(sigma))  # softplus for positivity ->Already changed to elu in model predictor
        # sigma = torch.clamp(sigma, min=self.eps, max=1e3)  # avoid 0 and inf
        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            #We get [N, C] but as C is 1 we can lose it

        total = 0.5 * torch.log(2 * torch.pi * (sigma**2 + self.eps)) + ((mu - y)**2 / (2 * (sigma**2 + self.eps)))
        return total.mean()
    
    def forward(self, mu, sigma, y, epoch):
        gnll = self.GNLL(mu, sigma, y)
        mse = self.MSE(mu, y)
        alpha = min(1.0, epoch / self.warm_up)
        gnll = alpha * gnll + (1 - alpha) * mse
        return gnll