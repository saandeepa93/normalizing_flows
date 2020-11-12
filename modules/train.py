# torch libraries
import torch 
from torch import nn, optim
from torch.distributions import MultivariateNormal

# python libraries
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# debugger
from sys import exit as e

# internal 
from modules.flows import Flow


def plot(arr):
  plt.scatter(arr[:, 0], arr[:, 1])
  plt.show()

def sample_moon(inpt):
  moon = make_moons(inpt, noise=0.05)[0].astype(np.float32)
  moon = torch.from_numpy(moon)
  return moon

def save_checkpoint(state, is_best, filename='./models/checkpoint.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, './models/model_best.pth.tar')



def test_data():
  dim = 2
  n_block = 9
  prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
  model = Flow(dim, prior, n_block)
  model.load_state_dict(torch.load("./models/model_best.pth.tar"))
  model.eval()
  xs = model.get_sample(1000)
  plot(xs.detach())


def train_data():
  inpt = 100
  dim = 2
  n_block = 9
  epochs = 1500
  lr = 0.01
  wd=1e-3
  old_loss = 1e6
  best_loss = 0
  prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
  # print(sample, torch.exp(prob), torch.exp(some_prob))
  

  model = Flow(dim, prior, n_block)
  optimizer = optim.Adam(model.parameters(), lr)

  for i in range(epochs):
    optimizer.zero_grad()
    d = sample_moon(inpt)
    d.requires_grad = True
    z, logdet, logprob = model(d)
    loss = -(logprob + logdet)
    loss = torch.sum(loss)
    if i % 100 == 0:
      if loss.item() < old_loss:
        is_best = 1
        save_checkpoint(model.state_dict(), is_best)
        old_loss = loss
      print(f"loss at epoch {i}: {loss.item()}")
    loss.backward()
    optimizer.step()
  best_loss = old_loss
  print(f"best loss at {best_loss}")





