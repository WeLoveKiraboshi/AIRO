import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np

import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from time import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cifar_dir = './dataset/cifar-10-batches-py'
for file in os.listdir(cifar_dir):
    print(file)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#1. Check the dataset
# what is the type of data that we have
# what is type label

data = unpickle(os.path.join(cifar_dir, 'data_batch_1'))
print(data.keys())
print(len(data[b'labels']))

data[b'labels'][:10]

data[b'data']
