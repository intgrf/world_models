from os.path import join
import gym
import numpy as np
import torch
from torchvision import transforms


def generate_data_fn(env_name, n_rollouts, path, policy):
    env = gym.make(env_name)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64))
    ])

    for rollout in range(n_rollouts):
        state = env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        dones = []

        while not done:
            action = policy(torch.from_numpy(state).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()
            state, reward, done, _ = env.step(action)
            state_rgb = env.render(mode='rgb_array')
            state_rgb = transform(state_rgb)
            state_rgb = np.asarray(state_rgb)
            #print(state_rgb.shape)
            states.append(state_rgb)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            if done:
                print('Rollout {}, length {}'.format(rollout, len(rewards)))
                np.savez(join(path, 'rollout_{}'.format(rollout)),
                         states=np.array(states),
                         rewards=np.array(rewards),
                         actions=np.array(actions),
                         dones=np.array(dones))
