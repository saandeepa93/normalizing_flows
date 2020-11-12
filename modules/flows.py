import torch 
from torch import nn

class SimpleNet(nn.Module):
  def __init__(self, inp, parity):
    super(SimpleNet, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(inp//2, 24),
      nn.ReLU(True),
      nn.Linear(24, 24),
      nn.ReLU(True),
      nn.Linear(24, inp//2),
      # nn.Sigmoid()
    )
    self.inp = inp
    self.parity = parity

  def forward(self, x):
    x0, x1 = x[:, ::2], x[:, 1::2]
    if self.parity % 2:
      x0, x1 = x1, x0 
    # print("X: ", x0[0].detach(), x1[0].detach())
    z1 = x1
    log_s = self.net(x1)
    t = self.net(x1)
    s = torch.exp(log_s)
    z0 = (s * x0) + t
    # print("Z: ", z0[0].detach(), z1[0].detach())
    if self.parity%2:
      z0, z1 = z1, z0
    z = torch.cat([z0, z1], dim = 1)
    logdet = torch.sum(torch.log(s), dim = 1)
    return z, logdet
  
  def reverse(self, z):
    z0, z1 = z[:, ::2], z[:, 1::2]
    if self.parity%2:
      z0, z1 = z1, z0
    # print("Z: ", z0[0].detach(), z1[0].detach())
    x1 = z1
    log_s = self.net(z1)
    t = self.net(z1)
    s = torch.exp(log_s)
    x0 = (z0 - t)/s
    # print("X: ", x0[0].detach(), x1[0].detach())
    if self.parity%2:
      x0, x1 = x1, x0
    x = torch.cat([x0, x1], dim = 1)
    return x
  

class Block(nn.Module):
  def __init__(self, inp, n_blocks):
    super(Block, self).__init__()
    parity = 0
    blocks = nn.ModuleList()
    for _ in range(n_blocks):
      blocks.append(SimpleNet(inp, parity))
      parity += 1
    self.blocks = blocks
  
  def forward(self, x):
    logdet = 0
    out = x
    xs = [out]
    # print("*"*20, "FORWARD", "*"*30)
    for block in self.blocks:
      out, det = block(out)
      logdet += det
      xs.append(out)
    return out, logdet

  def reverse(self, z):
    # print("*"*20, "REVERSE", "*"*30)
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
    logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
    return z, logdet, logprob
  
  def reverse(self, z):
    x = self.flow.reverse(z)
    return x
  
  def get_sample(self, n):
    z = self.prior.sample(sample_shape = torch.Size([n]))
    z_inv = self.reverse(z)
    return z_inv

