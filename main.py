# TODO:
# how to get state from mountaincar as image? -> set rgb mode instead of human mode
# implement VAE
#    how to sample from distribution?
# implement MDN-RNN model
import gym
from data_generator import generate_data_fn
import torch.nn.functional as F
import torch.nn as nn
import torch

def vae_loss(x, recon_x, mu, sigma):
    loss = F.mse_loss(recon_x, x, size_average=False)
    kld = -0.5 * torch.sum(1 + 2*sigma.log() - mu.pow(2) - sigma.pow(2))

    return loss + kld

action_size = 3
state_size = 2
hidden_size = 32

model = nn.Sequential(
        nn.Linear(state_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size))

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    model.load_state_dict(torch.load('./trained/trained_ddqn.data'))
    generate_data_fn(env_name='MountainCar-v0', n_rollouts=1, path='./rollout', policy=model)

