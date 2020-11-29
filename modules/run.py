import os
import time

import torch 
from torch import nn, optim
from torch.distributions import MultivariateNormal
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from modules.flows import Flow
from modules.train import train_data, test_data

def setup(rank, world_size):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()


def startup(rank, world_size, opt, use_cuda):
  print("on rank ", rank)
  torch.manual_seed(1)
  device = "cuda" if not opt.no_cuda and torch.cuda.is_available() else "cpu"
  setup(rank, world_size)
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  inpt = 100
  dim = 784
  n_block = 9
  epochs = 20
  lr = 0.001
  wd=1e-3
  old_loss = 1e6
  best_loss = 0
  batch_size = 128
  prior = MultivariateNormal(torch.zeros(dim).to(rank), torch.eye(dim).to(rank))

  #MNIST
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = MNIST(root=opt.root, train=True, transform=transform, \
    download=True)
  train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, \
    rank=rank)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, \
    num_workers=0, pin_memory=True, sampler=train_sampler)
  
  model = Flow(dim, prior, n_block)
  optimizer = optim.Adam(model.parameters(), lr)

  if use_cuda and torch.cuda.device_count()>1:
    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
  
  for epoch in range(epochs):
    t0 = time.time()
    model.train()
    train_data(opt, model, rank, train_loader, optimizer, epoch)
  print(f"time to complete {epochs} epoch: {time.time() - t0} seconds")
  cleanup()
