import torch 
from torch import nn

from sys import exit as e

class SimpleNet(nn.Module):
  def __init__(self, inp, parity):
    super(SimpleNet, self).__init__()
    self.net = nn.Sequential(
      nn.Conv2d(inp, 8, 3, 1, 1),
      nn.ReLU(True),
      nn.Conv2d(8, 16, 3, 1, 1),
      nn.ReLU(True),
      nn.Conv2d(16, inp, 3, 1, 1),
      nn.ReLU(True),
    )
    self.inp = inp
    self.parity = parity
  
  def forward(self, x):
    z = torch.zeros_like(x)
    x0, x1 = x[:, :, ::2, ::2], x[:, :, 1::2, 1::2]
    if self.parity % 2:
      x0, x1 = x1, x0 
    z1 = x1
    log_s = self.net(x1)
    t = self.net(x1)
    s = torch.exp(log_s)
    z0 = (s * x0) + t
    if self.parity%2:
      z0, z1 = z1, z0
    z[:, :, ::2, ::2] = z0
    z[:, :, 1::2, 1::2] = z1
    logdet = torch.sum(torch.log(s), dim = 1)
    return z, logdet
  
  def reverse(self, z):
    x = torch.zeros_like(z)
    z0, z1 = z[:, :, ::2, ::2], z[:, :, 1::2, 1::2]
    if self.parity%2:
      z0, z1 = z1, z0
    x1 = z1
    log_s = self.net(z1)
    t = self.net(z1)
    s = torch.exp(log_s)
    x0 = (z0 - t)/s
    if self.parity%2:
      x0, x1 = x1, x0
    x[:, :, ::2, ::2] = x0
    x[:, :, 1::2, 1::2] = x1
    return x
  

class Block(nn.Module):
  def __init__(self, inp, n_blocks):
    super(Block, self).__init__()
    parity = 0
    self.blocks = nn.ModuleList()
    for _ in range(n_blocks):
      self.blocks.append(SimpleNet(inp, parity))
      parity += 1
    
  
  def forward(self, x):
    logdet = 0
    out = x
    xs = [out]
    for block in self.blocks:
      out, det = block(out)
      logdet += det
      xs.append(out)
    return out, logdet

  def reverse(self, z):
    out = z
    for block in self.blocks[::-1]:
      out = block.reverse(out)
    return out


class Flow(nn.Module):
  def __init__(self, inp, prior, n_blocks):
    super(Flow, self).__init__()
    self.prior = prior
    self.flow = Block(inp, n_blocks)
  
  def forward(self, x):
    z, logdet = self.flow(x)
    logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1) #Error encountered here
    return z, logdet, logprob
  
  def reverse(self, z):
    x = self.flow.reverse(z)
    return x
  
  def get_sample(self, n):
    z = self.prior.sample(sample_shape = torch.Size([n]))
    return self.reverse(z)