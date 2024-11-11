import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """# Encoder Neural Network
    X_dim: Input dimension shape
    N: Inner neuronal layer size
    z_dim: Output dimension shape
    """

    def __init__(self, X_dim, N, z_dim, dropout=0.0, negative_slope=0.3):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3mu = nn.Linear(N, z_dim)
        self.lin3sigma = nn.Linear(N, z_dim)
        self.p = dropout
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.lin1(x)
        if self.p > 0.0:
            x = F.dropout(x, p=self.p, training=self.training)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)

        x = self.lin2(x)
        if self.p > 0.0:
            x = F.dropout(x, p=self.p, training=self.training)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)

        mu = self.lin3mu(x)
        sigma = self.lin3sigma(x)
        return mu, sigma
