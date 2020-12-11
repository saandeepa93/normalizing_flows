import os
import time

import torch 
from torch import nn, optim
from torch.distributions import MultivariateNormal
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.distributed as dist

from modules.flows import Flow
from modules.train import train_data, test_data

from sys import exit as e

def setup(rank, world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()


def startup(rank, world_size, opt, use_cuda):
  # print("on rank ", rank)
  torch.manual_seed(1)
  # device = "cuda" if not opt.no_cuda and torch.cuda.is_available() else "cpu"
  device= torch.device("cuda" if use_cuda else "cpu")
  # setup(rank, world_size)
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  inpt = 100
  dim = 784
  n_block = 9
  epochs = 20
  lr = 0.005
  wd=1e-3
  old_loss = 1e6
  best_loss = 0
  batch_size = 128
  prior = MultivariateNormal(torch.zeros(dim).to(device), torch.eye(dim).to(device))

  #MNIST
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = MNIST(root=opt.root, train=True, transform=transform, \
    download=True)
  # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, \
  #   rank=rank)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
    **kwargs)
  
  model = Flow(dim, prior, n_block)

  if use_cuda and torch.cuda.device_count()>1:
  #   model = model.to(rank)
    # model = DistributedDataParallel(model, device_ids=[rank])
    model = DataParallel(model, device_ids=[0, 1, 2, 3])

  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr)
  scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
  
  t0 = time.time()
  for epoch in range(epochs):
    model.train()
    train_data(opt, model, device, train_loader, optimizer, epoch)
    scheduler.step()
  print(f"time to complete {epochs} epoch: {time.time() - t0} seconds")
  # cleanup()
