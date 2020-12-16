# torch libraries
import torch 
from torch import nn, optim
from torchvision import utils
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



def plot_grad_flow(named_parameters, name, last):
  # path = f"./samples/{name}.png"
  path = f"/data/saandeepaath/flow_based/samples/{name}.png"
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
    plt.savefig(path)


def imshow(img, name=None):
  # path = f"./samples/{name}.png"
  path = f"/data/saandeepaath/flow_based/samples/{name}.png"
  # cv2.imshow("image", img)
  cv2.imwrite(path, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def save_checkpoint(state, is_best, filename='./models/checkpoint.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, './models/model_best.pth.tar')

def get_reverse(x, rank, model):
  x_1 = x[8:12].to(rank)
  x_new = model.module.reverse(x_1)
  img = x_1.view(4, 28, 28).detach().cpu().numpy()
  img_rev = x_new.view(4, 28, 28).detach().cpu().numpy()
  print(img.shape, img_rev.shape)
  for i in range(4):
    imshow(img[i] * 255, f"img_orig_{i}")
    imshow(img_rev[i] * 255, f"img_rev_{i}")

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


def train_data(opt, model, rank, train_loader, optimizer, epoch):
  path = "/data/saandeepaath/flow_based/samples/"
  z_sample = torch.randn((20, 1, 28, 28)).to(rank)
  for b, (x, _) in enumerate(train_loader):
    optimizer.zero_grad()
    x = x.to(rank)
    x = preprocess(x)
    z, logdet, logprob = model(x)
    # get_reverse(x, rank, model)
    logdet = logdet.mean()
    loss, logdet, logprob = calc_loss(logdet, logprob, 784)
    last = True if b == len(train_loader) - 1 else False
    loss.backward()
    plot_grad_flow(model.named_parameters(), f"grad_{epoch}_{b}", last)
    optimizer.step()
    if b % 10 == 0:
      dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      if rank == 0:
        with torch.no_grad():
          print("CUDA: ", z_sample.is_cuda)
          utils.save_image(
            model.reverse(z_sample).cpu().data,
            f"{path}/{str(b + 1).zfill(6)}.png",
            normalize=True,
            nrow=10,
            range=(-0.5, 0.5),
          )
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, \
          b * len(x), len(train_loader.dataset), 100. * b / \
            len(train_loader), loss.item()))




