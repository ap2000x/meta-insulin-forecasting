import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualMDN(nn.Module):
    """
    Probabilistic residual model using a Mixture Density Network (MDN)
    to predict a mixture of Gaussians over residuals.
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_gaussians=3):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.num_gaussians = num_gaussians
        self.pi = nn.Linear(hidden_dim, num_gaussians)        # mixture weights
        self.mu = nn.Linear(hidden_dim, num_gaussians)        # means
        self.sigma = nn.Linear(hidden_dim, num_gaussians)     # stddevs

    def forward(self, x):
        h = self.hidden(x)
        pi = F.softmax(self.pi(h), dim=-1)               # shape [batch, K]
        mu = self.mu(h)                                  # shape [batch, K]
        sigma = torch.exp(self.sigma(h)) + 1e-3          # shape [batch, K]
        return pi, mu, sigma

    def nll_loss(self, pi, mu, sigma, target):
        """Negative log likelihood loss for mixture of Gaussians"""
        m = torch.distributions.Normal(mu, sigma)
        probs = torch.exp(m.log_prob(target.unsqueeze(-1)))  # shape [batch, K]
        weighted = pi * probs
        loss = -torch.log(weighted.sum(dim=-1) + 1e-6)
        return loss.mean()

    def sample(self, pi, mu, sigma):
        """Sample from predicted Gaussian mixture"""
        B = pi.size(0)
        indices = torch.multinomial(pi, 1).squeeze()  # shape [B]
        selected_mu = mu[torch.arange(B), indices]
        selected_sigma = sigma[torch.arange(B), indices]
        return torch.normal(selected_mu, selected_sigma)