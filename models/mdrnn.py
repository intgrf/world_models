import torch
from torch import nn
import torch.nn.functional as F


class MDRNN(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size=256, n_gaussians=1, n_layers=1):
        super(MDRNN, self).__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.input_size = latent_size + action_size
        self.hidden_size = hidden_size
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers

        # self.rnn = nn.LSTM(input_size=self.input_size, num_layers=self.n_layers,
        #                    hidden_size=self.hidden_size, batch_first=True)
        self.rnn = nn.GRU(input_size=self.input_size, num_layers=self.n_layers,
                          hidden_size=self.hidden_size, batch_first=True)

        self.lin1 = nn.Linear(hidden_size, n_gaussians*latent_size)
        self.lin2 = nn.Linear(hidden_size, n_gaussians*latent_size)
        self.lin3 = nn.Linear(hidden_size, n_gaussians*latent_size)

    def get_mixture(self, x):
        n = x.size(1)
        pi, mu, sigma = self.lin1(x), self.lin2(x), self.lin3(x)

        pi = pi.view(-1, n, self.n_gaussians, self.latent_size)
        mu = mu.view(-1, n, self.n_gaussians, self.latent_size)
        sigma = sigma.view(-1, n, self.n_gaussians, self.latent_size)

        pi = F.softmax(pi, 2)
        sigma = torch.exp(sigma)
        return pi, mu, sigma

    def forward(self, latent, action):
        input = torch.cat((latent, action), dim=-1)
        output, _ = self.rnn(input)  # hidden = tuple (hidden_state, cell_state)
        pi, mu, sigma = self.get_mixture(output)

        return pi, mu, sigma

    # def init_hidden(self, device):
    #     return (torch.zeros(1, self.n_layers, self.hidden_size).to(device),
    #             torch.zeros(1, self.n_layers, self.hidden_size).to(device))

