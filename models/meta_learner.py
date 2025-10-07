import torch 
from torch import nn, optim
from copy import deepcopy

class MetaLearner:
    """
    Simple MAML-style meta-learning wrapper for personalization.
    Wraps a base model (e.g., ResidualMDN) and adapts it to new users quickly.
    """
    def __init__(self, model_class, model_kwargs, lr_inner=0.01, lr_outer=1e-3, device="cpu"):
        self.device = device
        self.base_model = model_class(**model_kwargs).to(device)
        self.lr_inner = lr_inner
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=lr_outer)

    def adapt(self, support_x, support_y):
        """
        Inner-loop adaptation step (on support set)
        Returns a fast-adapted copy of the model
        """
        model_clone = deepcopy(self.base_model)
        optimizer = optim.SGD(model_clone.parameters(), lr=self.lr_inner)
        model_clone.train()
        pi, mu, sigma = model_clone(support_x)
        loss = model_clone.nll_loss(pi, mu, sigma, support_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return model_clone

    def meta_update(self, tasks):
        """
        Outer-loop update across tasks
        Each task = (support_x, support_y, query_x, query_y)
        """
        meta_loss = 0.0
        self.optimizer.zero_grad()

        for support_x, support_y, query_x, query_y in tasks:
            adapted_model = self.adapt(support_x, support_y)
            pi, mu, sigma = adapted_model(query_x)
            loss = adapted_model.nll_loss(pi, mu, sigma, query_y)
            loss.backward()
            meta_loss += loss.item()

        self.optimizer.step()
        return meta_loss / len(tasks)

    def save(self, path):
        torch.save(self.base_model.state_dict(), path)

    def load(self, path):
        self.base_model.load_state_dict(torch.load(path, map_location=self.device))