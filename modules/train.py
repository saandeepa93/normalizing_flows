# torch libraries
import torch 
from torch import nn, optim

# distributed execution
import torch.distributed as dist

# python libraries
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import cv2
import time
from math import log

# debugger
from sys import exit as e

# internal 
from modules.flows import Flow

def init_weights(model):
  if type(model) == nn.Linear:
    torch.nn.init.xavier_uniform_(model.weight)
    model.bias.data.fill_(0.0)


def norm(AA, batch_size, height, width):
  AA = AA.view(AA.size(0), -1)
  AA -= AA.min(1, keepdim=True)[0]
  AA /= AA.max(1, keepdim=True)[0]
  AA = AA.view(batch_size, height, width)
  return AA

def plot(arr):
  plt.scatter(arr[:, 0], arr[:, 1])
  plt.show()

def show(img, name):
  # path = f"./samples/{name}.png"
  path = f"/data/saandeepaath/flow_based/samples/{name}.png"
  plt.imshow(img)
  # plt.show()
  plt.savefig(path)

def imshow(img):
  cv2.imshow("image", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def sample_moon(inpt):
  moon = make_moons(inpt, noise=0.05)[0].astype(np.float32)
  moon = torch.from_numpy(moon)
  return moon

def save_checkpoint(state, is_best, filename='./models/checkpoint.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, './models/model_best.pth.tar')

def preprocess(x):
  x = x * 255
  x = torch.floor(x/2**3)
  x = x/32 - 0.5
  return x

def calc_loss(logdet, logprob, num_pixels):
  loss = -log(32) * num_pixels
  loss = loss + logdet + logprob

  return (
    (-loss/(log(2) * num_pixels)).mean(),
    (logdet/(log(2) * num_pixels)).mean(),
    (logprob/(log(2) * num_pixels)).mean()
  )

def test_data():
  transform = transforms.Compose(
    [transforms.Normalize((0), (1))]
  )
  dim = 784
  n_block = 9
  prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
  model = Flow(dim, prior, n_block)
  model.load_state_dict(torch.load("./models/model_best.pth.tar"))
  model.eval()
  sample = prior.sample([1])
  sample = sample.view(28, 28).unsqueeze(0)
  sample = norm(sample, 1, 28, 28)
  xs = model.reverse(sample.view(1, 784))
  show(xs.view(28, 28).detach().numpy(), "output")


def train_data(opt, model, device, train_loader, optimizer, epoch):
  for b, (x, _) in enumerate(train_loader):
    optimizer.zero_grad()
    x = x.to(device)
    x = preprocess(x)
    x = x.view(x.size(0), -1)
    z, logdet, logprob = model.module(x)
    # x_1 = x[0:4]
    # x_new = model.module.reverse(x_1)
    # x_new = model.reverse(x_1)
    # img = x_1.view(4, 28, 28).detach().cpu().numpy()
    # img_rev = x_new.view(4, 28, 28).detach().cpu()
    # for i in range(4):
    #   show(img[i], f"img_orig_{i}")
    #   show(img_rev[i], f"img_rev_{i}")
    # print(np.amin(img), np.amax(img))
    # print(np.amin(img_rev), np.amax(img_rev))
    # loss = -(logprob + logdet)
    # loss = torch.sum(loss)
    logdet = logdet.mean()
    loss, logdet, logprob = calc_loss(logdet, logprob, 784)
    loss.backward()
    optimizer.step()
    if b % 10 == 0:
      # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, \
        b * len(x), len(train_loader.dataset), 100. * b / \
          len(train_loader), loss.item()))
        # print(f"epoch: {epoch} [{b}/{len(train_loader)} \
        # ({100. * b/len(train_loader)}%)]\t Loss:{loss.item()}")





