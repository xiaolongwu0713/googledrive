#### timm implementation
from __future__ import print_function

from prettytable import PrettyTable

from common_dl import set_random_seeds, count_parameters
import timm
import torch
from timm.models import registry
timm.list_models(pretrained=False)
model = timm.create_model('visformer_tiny',num_classes=5)

x     = torch.randn(1, 6, 224, 224) # can be any channel
model(x).shape
count_parameters(model)

registry.model_entrypoint('visformer_tiny') # find the class definition and change the input_size parameter in _cfg() function.
from timm.models.visformer import visformer_tiny




#####  ViT implementation

import glob
from itertools import chain
import os

import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
#from vit_pytorch.efficient import ViT # Effecient Attention
# for more info on x-former, general termed as the efficient transformer, check: https://arxiv.org/pdf/2107.02239v3.pdf
from vit_pytorch import ViT #vanilla vision transformer

# Training settings
batch_size = 64
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42



set_random_seeds(seed)

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

### change channels=6 for 6 input plans ####
linformer_model = ViT(dim=128,image_size=224,patch_size=32,num_classes=2,channels=6,transformer=efficient_transformer,)

### change the class __init__ function to have more plans ###
vit_model = ViT(image_size = 256,patch_size = 32,num_classes = 1000,dim = 1024,depth = 6,heads = 16,mlp_dim = 2048,dropout = 0.1,emb_dropout = 0.1)


x = torch.randn(1, 6, 224, 224) # can be any channels
model(x)
count_parameters(vit_model)