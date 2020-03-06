from models.vae import VAE
import gym
from PIL import Image
import torch
from torchvision import transforms

model = VAE(3, 32)

model.load_state_dict(torch.load('./trained/vae_32_policy_lr_1e-4_extra_5.data'))
env = gym.make('MountainCar-v0')
env.reset()
state = env.render(mode='rgb_array')
im = Image.fromarray(state)

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
im = transform(im).unsqueeze(0)

_, _, decoded = model(im)

decoded = decoded.squeeze(0)

toPIL = transforms.ToPILImage()

toPIL(decoded).save('decoded.png')
toPIL(im.squeeze(0)).save('real_input.png')