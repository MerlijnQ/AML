import torch
import torch.nn as nn

class customELBO(nn.Module):
    def __init__(self, total_epochs:int=100) -> None:
        """An ELBO loss used for a gaussian layer. Used for bayes by backprop implementations.

        Args:
            total_epochs (int, optional): Number of maximum epochs in training phase. Defaults to 100.
        """
        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.priors = {"mu_head": torch.tensor(1.0, device=self.device, dtype=torch.float32) ,
                       "sigma_head": torch.tensor(0.5, device=self.device, dtype=torch.float32)} #Only represent STD as we say mu=0 for both
        self.total_epochs = total_epochs
        self.eps = 1e-6
        
    def GNLL(self, mu:torch.tensor, sigma:torch.tensor, y:torch.tensor) -> float:
        """Gaussian Negative Log Likelihood (GNLL) loss, which takes both mu and sigma.

        Args:
            mu (torch.tensor): Prediction.
            sigma (torch.tensor): Predicted standard deviation.
            y (torch.tensor): Ground truth.

        Returns:
            float: GNLL loss
        """
        # Ensure sigma is valid
        sigma = torch.log1p(torch.exp(sigma))  # softplus for positivity
        sigma = torch.clamp(sigma, min=self.eps, max=1e3)  # avoid 0 and inf

        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            #We get [N, C] but as C is 1 we can lose it

        total = 0.5 * torch.log(2 * torch.pi * sigma**2) + ((mu - y)**2 / (2 * sigma**2 + self.eps))
        return total.mean()

    def KL(self, mu:torch.tensor, sigma:torch.tensor, sigma_p:torch.tensor)-> float:
        """KL divergence ment to penalize posterior distributon that differ a lot from the prior.

        Args:
            mu (torch.tensor): Prediction.
            sigma (torch.tensor): Predicted standard deviation.
            sigma_p (torch.tensor): Prior.

        Returns:
            float: KL divergence loss.
        """

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

    def warm_up_schedule(self, epoch:int, max_p:float)-> float:
        """Determines how much the bayesian layer is penalized with the KL divergence throughout training.

        Args:
            epoch (int): Current position in training. 
            max_p (float): Maximum scaler determining the max KL divergence scalcing.

        Returns:
            float: Penalty scaler.
        """
        #Warmup in 30% of epochs (arbitrarely chosen)
        return min(max_p, epoch / (0.3 * self.total_epochs))  

    def get_kl(self, m:nn.Module, sigma_p:torch.tensor, epoch:int, max_p:float)->float:
        """Determine the strength of the KL penalty per head.

        Args:
            m (nn.Module): Bayesian layer in the model.
            sigma_p (torch.tensor): Prior.
            epoch (int): Current position in training. 
            max_p (float): Maximum scaler determining the max KL divergence scalcing.

        Returns:
            float: Final KL divergence loss penalty for this epoch and forward pass.
        """
         
        #To allow for warm up we want the model to explore first.
        #Therefore we increase the KL penalty with time per head.
        #Max penalty for sigma should be lower to prevent variance collapse in the model.
        penalty = self.warm_up_schedule(epoch, max_p)

        kl1 = self.KL(m.W_mu, m.W_sigma, sigma_p)
        kl2 = self.KL(m.b_mu, m.b_sigma, sigma_p)
        return (kl1 + kl2) * penalty
    
    def forward(self,
                 mu:torch.tensor, sigma:torch.tensor, y:torch.tensor, model:nn.Module, epoch:int, train:bool=True
                 ) -> float:
        """_summary_

        Args:
            mu (torch.tensor): Prediciton.
            sigma (torch.tensor): Standard deviation with prediction.
            y (torch.tensor): Ground truth.
            model (nn.Module): PyTorch model.
            epoch (int): Current position in training. 
            train (bool, optional): Determines wether the model is in the training phase. Defaults to True.

        Returns:
            float: Final model loss.
        """
        gnll = self.GNLL(mu, sigma, y)
        if train:
            kl1 = self.get_kl(model.mu, self.priors["mu_head"], epoch, max_p=1)
            kl2 = self.get_kl(model.sigma, self.priors["sigma_head"], epoch, max_p=0.1)
            total_kl = kl1 + kl2
            return gnll + total_kl
        return gnll
    
class gnll(nn.Module):
    def __init__(self, warm_up:int=10) -> None:
        """Initialize loss criterion

        Args:
            warm_up (int, optional): Warm-up phase duration expressed in epochs. Defaults to 10.
        """

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = 1e-6
        self.warm_up = warm_up

    def MSE(self, mu:torch.tensor, y:torch.tensor) -> float:
        """Mean squared error (MSE) loss

        Args:
            mu (torch.tensor): Prediction.
            y (torch.tensor): Ground truth. 

        Returns:
            float: MSE loss
        """

        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
        mse = torch.mean((mu - y) ** 2)
        return mse
        
    def GNLL(self, mu:torch.tensor, sigma:torch.tensor, y:torch.tensor) -> float:
        """Gaussian Negative Log Likelihood (GNLL) loss, which takes both mu and sigma.

        Args:
            mu (torch.tensor): Prediction.
            sigma (torch.tensor): Predicted standard deviation.
            y (torch.tensor): Ground truth.

        Returns:
            float: GNLL loss
        """

        if mu.dim() == 2:
            mu = mu.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
            #We get [N, C] but as C is 1 we can lose it

        total = 0.5 * torch.log(2 * torch.pi * (sigma**2 + self.eps)) + ((mu - y)**2 / (2 * (sigma**2 + self.eps)))
        return total.mean()
    
    def forward(self, mu:torch.tensor, sigma:torch.tensor, y:torch.tensor, epoch:int) -> float:
        """A forward pass to calculate the loss. This sclaes between the MSE and GNLL depending
          on where the training process is (with respect to the warm-up phase).

        Args:
            mu (torch.tensor): Predictions.
            sigma (torch.tensor): Standard deviations.
            y (torch.tensor): Ground truths.
            epoch (int): Epoch determining where the model is in the training phase.

        Returns:
            float: Final loss for this epoch and data.
        """

        gnll = self.GNLL(mu, sigma, y)
        mse = self.MSE(mu, y)
        alpha = min(1.0, epoch / self.warm_up) #Linear warm-up
        
        #Cos warm-up (increases faster later on)
        # t = min(epoch / self.warm_up, 1.0)
        # alpha = 0.5 * (1 - math.cos(math.pi * t)) 

        gnll = alpha * gnll + (1 - alpha) * mse
        return gnll