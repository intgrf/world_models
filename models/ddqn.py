import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from memory import Memory
import random
import copy


class DDQN(nn.Module):
    memory_capacity = 5000
    batch_size = 128
    max_step = 70001
    update_step = 1000

    max_eps = 0.5
    min_eps = 0.1

    gamma = 0.99
    lr = 0.00003

    def __init__(self, model, target_model, env, state_size, action_size):
        super(DDQN, self).__init__()
        self.device = torch.device('cuda')

        self.env = env
        self.action_size = action_size
        self.state_size = state_size

        self.model = model
        self.target_model = target_model

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.memory = Memory(self.memory_capacity)
        self.writer = SummaryWriter('./result/mc/mountain-car-2')

    def select_action(self, epsilon, state, exploit=False):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        if exploit:
            return self.target_model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1,1).item()
        else:
            return self.model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def exploit(self, render=True):
        state = self.env.reset()
        r = 0.
        is_terminal = False

        while not is_terminal:
            eps = 0.
            if render:
                self.env.render()
            action = self.select_action(eps, state, exploit=True)
            state, reward, is_terminal, _ = self.env.step(action)
            r += reward
        return r

    def optimize(self, batch):
        state, action, reward, next_state, done = batch

        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)
        with torch.no_grad():
            target_q[done] = self.target_model(next_state).max(1)[0].detach()[done]
        target_q = reward + target_q * self.gamma

        q = self.model(state).gather(1, action.unsqueeze(1))

        loss = nn.functional.smooth_l1_loss(q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self):
        state = self.env.reset()
        current_episode = 0
        for step in range(self.max_step):
            #print(step)
            # select action
            eps = self.max_eps - (self.max_eps - self.min_eps) * step / self.max_step
            action = self.select_action(eps, state)

            # execute action in emulator
            new_state, reward, done, _ = self.env.step(action)

            reward = reward + 300 * (self.gamma * abs(new_state[1]) - abs(state[1]))
            # store info in replay memory
            self.memory.append((state, action, reward, new_state, done))

            # update state
            if done:
                state = self.env.reset()
                done = False
                current_episode += 1
                print(current_episode)
            else:
                state = new_state

            # make gradient descent step
            # sample random minibatch from replay memory
            if step > self.batch_size:
                self.optimize(list(zip(*self.memory.sample(self.batch_size))))

            # update target model
            if step % self.update_step == 0:
                self.update_target_model()
                r = self.exploit(render=False)
                self.writer.add_scalar('exploitation_reward', r, global_step=current_episode)

