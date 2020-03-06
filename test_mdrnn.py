import torch
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models.vae import VAE
from models.mdrnn import MDRNN

import gym

# + generate state + action + next_state
# encode state with vae
# predict next_state with mdrnn
# decode with vae
# compare

env = gym.make('MountainCar-v0')

# define vae
VAE_PATH = './trained/vae_32_policy_lr_1e-4_extra_5.data'
LATENT_SIZE = 32

vae = VAE(3, LATENT_SIZE)
vae.load_state_dict(torch.load(VAE_PATH))

# define mdrnn
MDRNN_PATH = './trained/mdrnn/fix_loss_epoch_140.data'
mdrnn = MDRNN(latent_size=LATENT_SIZE, action_size=1, n_gaussians=5)
mdrnn.load_state_dict(torch.load(MDRNN_PATH))

# define policy

state_size = 2
action_size = 3
hidden_size = 32
policy = nn.Sequential(
        nn.Linear(state_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_size))

policy.load_state_dict(torch.load('./trained/trained_ddqn.data'))

s = env.reset()
state = env.render(mode='rgb_array')
action = policy(torch.from_numpy(s).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()
next_s = env.step(action)
next_state = env.render(mode='rgb_array')

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((64, 64)),
                                transforms.ToTensor()])

state = transform(state).unsqueeze(0)
mu, sigma, _ = vae(state)
y = torch.randn_like(sigma)
state_enc = y.mul(sigma).add_(mu).unsqueeze(0)

action = torch.tensor([action]).unsqueeze(0).unsqueeze(0).float()

pi, mu, sigma = mdrnn(state_enc, action)

next_state_pred = torch.normal(mu, sigma)
pred = torch.sum(next_state_pred * pi, dim=2)
'./trained/mdrnn/second_try.data'
decoded_pred = vae.decoder(pred.squeeze(0)).squeeze(0)

toPIL = transforms.ToPILImage()
# mean_vec = torch.mean(decoded_pred, dim=0)
# for i in range(5):
#     toPIL(decoded_pred[i]).save('rnn_pred_gauss_5_{}.png'.format(i))
toPIL(state.squeeze(0)).save('first.png')
toPIL(decoded_pred).save('pred.png')
# toPIL(mean_vec).save('rnn_pred_mean.png')

im = transform(next_state)
toPIL(im).save('real.png')

