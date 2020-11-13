# torch libraries
import torch 
from torch import nn, optim
from torch.distributions import MultivariateNormal

#MNIST
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

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

def show(img):
  plt.imshow(img)
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
  dim = 784
  n_block = 1
  epochs = 1500
  lr = 0.01
  wd=1e-3
  old_loss = 1e6
  best_loss = 0
  batch_size = 1
  prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

  #MNIST
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  model = Flow(dim, prior, n_block)
  optimizer = optim.Adam(model.parameters(), lr)

  for i in range(epochs):
    total_loss = 0
    for b, (x, _) in enumerate(train_loader):
      x = x.view(batch_size, -1)
      optimizer.zero_grad()
      z, logdet, logprob = model(x)
      xs = model.reverse(z)
      img_orig = x.view(batch_size, 28, 28)
      img_test = z.reshape(batch_size, 28, 28)
      loss = -(logprob + logdet)
      loss = torch.sum(loss)
      total_loss += loss
    if i % 1 == 0:
      if loss.item() < old_loss:
        is_best = 1
        save_checkpoint(model.state_dict(), is_best)
        old_loss = loss
      print(f"loss at epoch {i}: {loss.item()}")
    loss.backward()
    optimizer.step()
  best_loss = old_loss
  print(f"best loss at {best_loss}")





