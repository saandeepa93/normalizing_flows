import argparse
import torch 
from sys import exit as e

from modules.train import train_data, test_data


def main():
  device = 'gpu' if torch.cuda.is_available() else 'cpu'
  print(device)
  train_data(device)
  test_data(device)


if __name__ == '__main__':
  main()