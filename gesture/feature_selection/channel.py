import sys
import socket
if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
from gesture.config import *

import os, re
import matplotlib.pyplot as plt
import hdf5storage
import numpy as np
import torch
import random
from common_dl import set_random_seeds
from common_dl import myDataset
from comm_utils import slide_epochs
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from gesture.models.deepmodel import deepnet,deepnet_resnet
from example.gumbelSelection.ChannelSelection.models import MSFBCNN
from gesture.models.selectionModels import selectionNet

from gesture.myskorch import on_epoch_begin_callback, on_batch_end_callback
from gesture.preprocess.chn_settings import get_channel_setting

result_dir='/Users/long/Documents/data/gesture/training_result/selection/P10/gumbel'
ZZ=np.load(result_dir+'/ZZ.npy')
plt.imshow(ZZ[80][0])












