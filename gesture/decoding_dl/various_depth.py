'''
deepconv decoding performance with different depth
'''
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import *
from natsort import natsorted,realsorted
from common_plot import barplot_annotate_brackets

good_sid=[4,10,13,41]
depths=[1,2,3,4,5,6]
data_dir = '/Users/long/Documents/data/gesture/'# temp data dir
training_result_dir=data_dir+'training_result/dl_depth/'

accuracy_all=[]
for sid in good_sid:
    sid_acc=[]
    tmp = realsorted([pth for pth in Path(training_result_dir+str(sid)).iterdir() if pth.suffix == '.npy' and 'changeDepth' in str(pth)])
    for depth in depths:
        result = np.load(str(tmp[depth-1]), allow_pickle=True).item()
        sid_acc.append(result['test_acc'])
    accuracy_all.append(sid_acc)
# perform best at depth=1
accuracies=[]

