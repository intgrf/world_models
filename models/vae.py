import torch
from torch import nn
import  torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.channels = channels

        self.conv1 = nn.Conv2d(channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_sigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, channels, latent_size):
        super(Decoder, self).__init__()
        self.channels = channels
        self.latent_size = latent_size

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, channels, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        decoded = torch.sigmoid(self.deconv4(x))
        return decoded


class VAE(nn.Module):
    def __init__(self, channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(channels, latent_size)
        self.decoder = Decoder(channels, latent_size)

    def forward(self, x):
        mu, sigma = self.encoder(x)
        sigma = sigma.exp()
        # sample y from normal distribution with same size as sigma
        y = torch.randn_like(sigma)
        z = y.mul(sigma).add_(mu)

        decoded = self.decoder(z)
        return mu, sigma, decoded

