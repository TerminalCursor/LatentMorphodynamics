import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid


class Decoder(nn.Module):
    """# Decoder Neural Network
    X_dim: Output dimension shape
    N: Inner neuronal layer size
    z_dim: Input dimension shape
    """

    def __init__(self, X_dim, N, z_dim, dropout=0.0, negative_slope=0.3):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
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

        x = self.lin3(x)
        return sigmoid(x)
