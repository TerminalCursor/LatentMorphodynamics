import torch
from torch import nn


class ConvolutionalVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, hidden_dims, image_shape, dropout=0.0):
        super(ConvolutionalVariationalAutoencoder, self).__init__()

        self.latent_dims = latent_dims  # Size of the latent space layer
        self.hidden_dims = (
            hidden_dims  # List of hidden layers number of filters/channels
        )
        self.image_shape = image_shape  # Input image shape

        self.last_channels = self.hidden_dims[-1]
        self.in_channels = self.image_shape[0]
        # Simple formula to get the number of neurons after the last convolution layer is flattened
        self.flattened_channels = int(
            self.last_channels
            * (self.image_shape[1] / (2 ** len(self.hidden_dims))) ** 2
        )

        # For each hidden layer we will create a Convolution Block
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout),
                )
            )

            self.in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Here are our layers for our latent space distribution
        self.fc_mu = nn.Linear(self.flattened_channels, latent_dims)
        self.fc_var = nn.Linear(self.flattened_channels, latent_dims)

        # Decoder input layer
        self.decoder_input = nn.Linear(latent_dims, self.flattened_channels)

        # For each Convolution Block created on the Encoder we will do a symmetric Decoder with the same Blocks, but using ConvTranspose
        self.hidden_dims.reverse()
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=self.in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout),
                )
            )

            self.in_channels = h_dim

        self.decoder = nn.Sequential(*modules)

        # The final layer the reconstructed image have the same dimensions as the input image
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.image_shape[0],
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def get_latent_dims(self):

        return self.latent_dims

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var componentsbof the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes onto the image space.
        """
        result = self.decoder_input(z)
        result = result.view(
            -1,
            self.last_channels,
            int(self.image_shape[1] / (2 ** len(self.hidden_dims))),
            int(self.image_shape[1] / (2 ** len(self.hidden_dims))),
        )
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, input):
        """
        Forward method which will encode and decode our image.
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), input, mu, log_var, z]

    def loss_function(self, recons, input, mu, log_var):
        """
        Computes VAE loss function
        """
        recons_loss = nn.functional.binary_cross_entropy(
            recons.reshape(recons.shape[0], -1),
            input.reshape(input.shape[0], -1),
            reduction="none",
        ).sum(dim=-1)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        loss = (recons_loss + kld_loss).mean(dim=0)

        return loss

    def sample(self, num_samples, device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        """
        z = torch.randn(num_samples, self.latent_dims)
        z = z.to(device)
        samples = self.decode(z)

        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)[0]

    def interpolate(self, starting_inputs, ending_inputs, device, granularity=10):
        """This function performs a linear interpolation in the latent space of the autoencoder
        from starting inputs to ending inputs. It returns the interpolation trajectories.
        """
        mu, log_var = self.encode(starting_inputs.to(device))
        starting_z = self.reparameterize(mu, log_var)

        mu, log_var = self.encode(ending_inputs.to(device))
        ending_z = self.reparameterize(mu, log_var)

        t = torch.linspace(0, 1, granularity).to(device)

        intep_line = torch.kron(
            starting_z.reshape(starting_z.shape[0], -1), (1 - t).unsqueeze(-1)
        ) + torch.kron(ending_z.reshape(ending_z.shape[0], -1), t.unsqueeze(-1))

        decoded_line = self.decode(intep_line).reshape(
            (starting_inputs.shape[0], t.shape[0]) + (starting_inputs.shape[1:])
        )
        return decoded_line
