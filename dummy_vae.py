import torch
import torch.utils.data
from torchvision.datasets.folder import ImageFolder
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.vae import VAE
import gym
from PIL import Image
from datasets import ObservationDataset
# from dataclasses import dataclass
# from typing import List, Union
# from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

SIZE = 64
LATENT_SIZE = 32
BATCH_SIZE = 128
device = torch.device('cuda')

model = VAE(3, LATENT_SIZE)
#model.load_state_dict(torch.load('./trained/vae_latent64_epoch_extra10.data'))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, 'min')
writer = SummaryWriter('./result/vae/train_64_buf30_2')

model.to(device)


def loss_function(x, decoded_x, mu, sigma):
    loss = F.mse_loss(decoded_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + 2*sigma - mu.pow(2) - (2*sigma).exp())

    return loss + KLD


def train(epoch):
    step = 0
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.permute(0, 2, 3, 1)
        batch = batch.to(device)
        #print(len(batch))
        optimizer.zero_grad()
        mu, sigma, decoded_batch = model(batch)
        loss = loss_function(batch, decoded_batch, mu, sigma)
        train_loss += loss.item()
        #writer.add_scalar('train_loss', loss.item(), global_step=step)
        loss.backward()
        optimizer.step()

        print('step: {}, loss {}'.format(step, loss.item()))
        step += 1
    print('---> train epoch {}, avg loss {}'.format(epoch, train_loss/step))
    writer.add_scalar('train_loss', train_loss/len(train_loader.dataset), global_step=epoch)


def test(epoch):
    test_loss = 0
    test_dataset.load_next_buffer()
    with torch.no_grad():
        step = 0
        for batch_idx, batch in enumerate(test_loader):
            print('step {}'.format(step))
            batch = batch.permute(0, 2, 3, 1)
            batch = batch.to(device)
            #print(len(batch))
            mu, sigma, decoded_batch = model(batch)
            test_loss += loss_function(batch, decoded_batch, mu, sigma).item()
            step += 1
    #writer.add_scalar('test_loss', test_loss, global_step=epoch)
    print('---> test epoch {}, avg loss {}'.format(epoch, test_loss/step))
    writer.add_scalar('test_loss', test_loss/len(test_loader.dataset), global_step=epoch)
    return test_loss/step


for epoch in range(15):
    train(epoch)
    l = test(epoch)
    scheduler.step(l)

#model.load_state_dict(torch.load('./trained/vae_latent64_epoch10.data'))
# env = gym.make('MountainCar-v0')
# env.reset()
# state = env.render(mode='rgb_array')
# im = Image.fromarray(state)
# im = transform_test(im).unsqueeze(0)
#
# _, _, decoded = model(im)
#
# decoded = decoded.squeeze(0)
#
# toPIL = transforms.ToPILImage()
#
# toPIL(decoded).save('decoded_extra10.png')
# toPIL(im.squeeze(0)).save('input_extra10 .png')