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
from  matplotlib.lines import Line2D
from sklearn.datasets import make_moons
import cv2
import time
from math import log

# debugger
from sys import exit as e

# internal 
from modules.flows import Flow


def plot_grad_flow(named_parameters, name, last):
  # path = f"./samples/{name}.png"
  path = f"/data/saandesepaath/flow_based/samples/{name}.png"
  ave_grads = []
  layers = []
  for n, p in named_parameters:
    if(p.requires_grad) and ("bias" not in n):
      layers.append(n)
      ave_grads.append(p.grad.abs().mean())
  plt.plot(ave_grads, alpha=0.3, color="b")
  # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(xmin=0, xmax=len(ave_grads))
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
  if last:
    plt.savefig(name)

def plot_grad_flow_2(named_parameters, name):
  '''Plots the gradients flowing through different layers in the net during training.
  Can be used for checking for possible gradient vanishing / exploding problems.
  
  Usage: Plug this function in Trainer class after loss.backwards() as 
  "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
  # path = f"./samples/{name}.png"
  path = f"/data/saandeepaath/flow_based/samples/{name}.png"
  ave_grads = []
  max_grads= []
  layers = []
  for n, p in named_parameters:
    if(p.requires_grad) and ("bias" not in n):
      layers.append(n)
      ave_grads.append(p.grad.abs().mean())
      max_grads.append(p.grad.abs().max())
  plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
  plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(left=0, right=len(ave_grads))
  plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
  plt.legend([Line2D([0], [0], color="c", lw=4),
              Line2D([0], [0], color="b", lw=4),
              Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
  plt.savefig(path)



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

def show(img, name=None):
  # path = f"./samples/{name}.png"
  # path = f"/data/saandeepaath/flow_based/samples/{name}.png"
  plt.imshow(img)
  # plt.show()
  plt.savefig(path)

def imshow(img, name=None):
  # path = f"./samples/{name}.png"
  # path = f"/data/saandeepaath/flow_based/samples/{name}.png"
  # cv2.imshow("image", img)
  cv2.imwrite(path, img)
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
  dim = 1
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
    z, logdet, logprob = model.forward(x)


    # x_1 = x[8:12]
    # # x_new = model.module.reverse(x_1)
    # x_new = model.reverse(x_1)
    # img = x_1.view(4, 28, 28).detach().cpu().numpy()
    # img_rev = x_new.view(4, 28, 28).detach().cpu().numpy()
    # print(img.shape, img_rev.shape)
    # for i in range(4):
    #   imshow(img[i] * 255, f"img_orig_{i}")
    #   imshow(img_rev[i] * 255, f"img_rev_{i}")
    # # print(np.amin(img), np.amax(img))
    # # print(np.amin(img_rev), np.amax(img_rev))
    # e()

    # loss = -(logprob + logdet)
    # loss = torch.sum(loss)
    logdet = logdet.mean()
    loss, logdet, logprob = calc_loss(logdet, logprob, 784)
    loss.backward()
    last = True if b == len(train_loader) - 1 else False
    plot_grad_flow(model.named_parameters(), f"grad_{epoch}_{b}", last)
    optimizer.step()
    if b % 10 == 0:
      # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, \
        b * len(x), len(train_loader.dataset), 100. * b / \
          len(train_loader), loss.item()))
        # print(f"epoch: {epoch} [{b}/{len(train_loader)} \
        # ({100. * b/len(train_loader)}%)]\t Loss:{loss.item()}")




