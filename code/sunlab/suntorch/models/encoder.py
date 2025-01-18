import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """# Encoder Neural Network
    X_dim: Input dimension shape
    N: Inner neuronal layer size
    z_dim: Output dimension shape
    """

    def __init__(self, X_dim, N, z_dim, depth=2, relu_type=1, dropout=0.0, negative_slope=0.3):
        super(Encoder, self).__init__()
        assert relu_type in [0,1,2]
        assert depth>=1
        assert depth<=4
        self.depth = depth-1
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2_1 = nn.Linear(N, N)
        self.lin2_2 = nn.Linear(N, N)
        self.lin2_3 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
        self.p = dropout
        self.negative_slope = negative_slope
        self.relu_type = relu_type
        if relu_type == 0:
            self.relu = F.relu
        elif relu_type == 1:
            self.relu = F.leaky_relu
        elif relu_type == 2:
            self.relu = F.elu

    def forward(self, x):
        x = self.lin1(x)
        if self.p > 0.0:
            x = F.dropout(x, p=self.p, training=self.training)
        if self.relu_type == 1:
            x = self.relu(x, negative_slope=self.negative_slope)
        else:
            x = self.relu(x)

        if self.depth >= 1:
            x = self.lin2_1(x)
            if self.p > 0.0:
                x = F.dropout(x, p=self.p, training=self.training)
            if self.relu_type == 1:
                x = self.relu(x, negative_slope=self.negative_slope)
            else:
                x = self.relu(x)

        if self.depth >= 2:
            x = self.lin2_2(x)
            if self.p > 0.0:
                x = F.dropout(x, p=self.p, training=self.training)
            if self.relu_type == 1:
                x = self.relu(x, negative_slope=self.negative_slope)
            else:
                x = self.relu(x)

        if self.depth == 3:
            x = self.lin2_3(x)
            if self.p > 0.0:
                x = F.dropout(x, p=self.p, training=self.training)
            if self.relu_type == 1:
                x = self.relu(x, negative_slope=self.negative_slope)
            else:
                x = self.relu(x)

#         x = self.lin2(x)
#         if self.p > 0.0:
#             x = F.dropout(x, p=self.p, training=self.training)
#         x = F.leaky_relu(x, negative_slope=self.negative_slope)

        xgauss = self.lin3gauss(x)
        return xgauss
