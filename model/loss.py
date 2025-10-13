import torch

class customELBO():
    def __init__(self, total_epochs=50):
        self.priors = {"mu_head":1.0 ,
                       "sigma_head":0.5} #Only represent STD as we say mu=0 for both
        self.total_epochs = total_epochs
        
    def GNLL(self, mu, sigma, y):

        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            #We get [N, C] but as C is 1 we can lose it

        total = (0.5 * torch.log(2 * torch.pi * sigma) + ((mu - y)**2 / (2 * sigma) ))
        return total.mean()

    def KL(self, mu, sigma, sigma_p):
        kl = torch.log(sigma_p/sigma) + (sigma**2 + mu**2)/(2 * sigma_p**2) - 0.5
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
    
    def forward(self, mu, sigma, y, model, epoch):
        gnll = self.GNLL(mu, sigma, y)
        kl1 = self.get_kl(model.mu, self.priors["mu_head"], epoch, max_p=1)
        kl2 = self.get_kl(model.sigma, self.priors["sigma_head"], epoch, max_p=0.1)
        total_kl = kl1+  kl2
        return gnll + self.penalty * total_kl