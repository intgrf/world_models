from os.path import join
import gym
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def generate_data_fn(env_name, n_rollouts, path, policy):
    env = gym.make(env_name)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.Resize((64, 64))
    ])

    for rollout in range(n_rollouts):
        # if rollout > n_rollouts*0.7:
        #     path = './mountaincar_states/test/first'
        state = env.reset()
        done = False
        frame = 0
        while not done:
            action = policy(torch.from_numpy(state).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()
            state, _, done, _ = env.step(action)
            state_rgb = env.render(mode='rgb_array')
            state_rgb = transform(state_rgb)
            state_rgb.save(join(path, 'rollout_{}_frame_{}'.format(rollout, frame)) + '.jpg')
            frame += 1
