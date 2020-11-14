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
import cv2

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
  plt.imshow(img)
  # plt.show()
  plt.savefig(f"./samples/{name}.png")

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



def test_data():
  transform = transforms.Compose(
    [transforms.Normalize((0), (1))]
  )
  dim = 784
  n_block = 9
  prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))
  model = Flow(dim, prior, n_block)
  model.load_state_dict(torch.load("./models/mnist.pth.tar"))
  model.eval()
  sample = prior.sample([1])
  xs  = model.get_sample(1)
  sample = sample.view(28, 28).unsqueeze(0)
  xs = norm(xs, 1, 28, 28).squeeze()
  print(torch.min(xs), torch.max(xs))
  show(sample.view(28, 28).detach().numpy())
  show(xs.view(28, 28).detach().numpy())
  # imshow(z.view(28, 28).detach().numpy())
  # imshow(sample.view(28, 28).detach().numpy())
  # plot(xs.detach())

def train_data():
  inpt = 100
  dim = 784
  n_block = 9
  epochs = 150
  lr = 0.001
  wd=1e-3
  old_loss = 1e6
  best_loss = 0
  batch_size = 100
  prior = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

  #MNIST
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  torch.manual_seed(1)
  model = Flow(dim, prior, n_block)
  model.apply(init_weights)
  optimizer = optim.Adam(model.parameters(), lr)

  sample = prior.sample([1])
  for i in range(epochs):
    total_loss = 0
    for b, (x, _) in enumerate(train_loader):
      model.train()
      optimizer.zero_grad()
      if b == 100:
        break
      x = x.view(batch_size, -1)
      z, logdet, logprob = model(x)
      img_orig = x.reshape(batch_size, 28, 28)
      img_test = model.reverse(z).reshape(batch_size, 28, 28)
      # imshow(img_orig[0].detach().numpy())
      # imshow(img_test[0].detach().numpy())
      # e()
      loss = -(logprob + logdet)
      loss = torch.sum(loss)
      total_loss += loss
    if i % 10 == 0:
      model.eval()
      # for params in model.parameters():
      #   print(params.mean())
      if loss.item()/batch_size < old_loss and loss.item() > 0:
        is_best = 1
        save_checkpoint(model.state_dict(), is_best)
        print("saved model")
        old_loss = loss/batch_size
        xs = model.reverse(sample)
        show(xs.squeeze().view(28, 28).detach().numpy(), i)
        # show(x[0].squeeze().view(28, 28).detach().numpy(), i)
      print(f"loss at epoch {i}: {loss.item()/batch_size}")
    model.train()
    loss.backward()
    optimizer.step()
  best_loss = old_loss
  print(f"best loss at {best_loss}")





