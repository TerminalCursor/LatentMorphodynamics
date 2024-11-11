import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .encoder import Encoder
from .decoder import Decoder
from .common import *


class Autoencoder:
    """# Autoencoder Model"""

    def __init__(
        self, data_dim, latent_dim, enc_dec_size, negative_slope=0.3, dropout=0.0
    ):
        self.encoder = Encoder(
            data_dim,
            enc_dec_size,
            latent_dim,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.decoder = Decoder(
            data_dim,
            enc_dec_size,
            latent_dim,
            negative_slope=negative_slope,
            dropout=dropout,
        )
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.p = dropout
        self.negative_slope = negative_slope
        return

    def parameters(self):
        return (*self.encoder.parameters(), *self.decoder.parameters())

    def train(self):
        self.encoder.train(True)
        self.decoder.train(True)
        return self

    def eval(self):
        self.encoder.train(False)
        self.decoder.train(False)
        return self

    def encode(self, data):
        return self.encoder(data)

    def decode(self, data):
        return self.decoder(data)

    def __call__(self, data):
        return self.decode(self.encode(data))

    def save(self, base="./"):
        torch.save(self.encoder.state_dict(), base + "weights_encoder.pt")
        torch.save(self.decoder.state_dict(), base + "weights_decoder.pt")
        return self

    def load(self, base="./"):
        self.encoder.load_state_dict(torch.load(base + "weights_encoder.pt"))
        self.encoder.eval()
        self.decoder.load_state_dict(torch.load(base + "weights_decoder.pt"))
        self.decoder.eval()
        return self

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.discriminator.to(device)
        return self

    def cuda(self):
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        return self

    def cpu(self):
        self.encoder = self.encoder.cpu()
        self.decoder = self.decoder.cpu()
        return self

    def init_optimizers(self, recon_lr=1e-4):
        self.optim_E_enc = torch.optim.Adam(self.encoder.parameters(), lr=recon_lr)
        self.optim_D = torch.optim.Adam(self.decoder.parameters(), lr=recon_lr)
        return self

    def init_losses(self, recon_loss_fn=F.binary_cross_entropy):
        self.recon_loss_fn = recon_loss_fn
        return self

    def train_step(self, raw_data):
        data = to_var(raw_data.view(raw_data.size(0), -1))

        self.encoder.zero_grad()
        self.decoder.zero_grad()
        z = self.encoder(data)
        X = self.decoder(z)
        self.recon_loss = self.recon_loss_fn(X + EPS, data + EPS)
        self.recon_loss.backward()
        self.optim_E_enc.step()
        self.optim_D.step()
        return

    def losses(self):
        try:
            return self.recon_loss
        except:
            ...
        return
