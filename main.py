import argparse
import torch 
import os
from sys import exit as e

import torch.multiprocessing as mp

from modules.run import startup

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--no-cuda', default=False, help="disables CUDA if true")
  parser.add_argument("--gpus", default=4, help="# of GPUS in a node")
  parser.add_argument("--root", default="./data")
  parser.add_argument("--local_rank", default=0, type=int)
  opt = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
  
  use_cuda = not opt.no_cuda and torch.cuda.is_available()
  world_size = opt.gpus if use_cuda else 2
  mp.spawn(startup, args=(world_size, opt, use_cuda), nprocs=world_size, join=True)
  # test_data()


if __name__ == '__main__':
  main()