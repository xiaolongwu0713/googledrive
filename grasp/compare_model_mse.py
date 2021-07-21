'''
compare the mse between different model and plot the mse error bar.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from grasp.config import *
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42


sid=10
movements=4
models=['linear','pls','ukf','TSception','deepConv','shallowConv']
input='frequency_and_raw' # for TSception

plot_dir = data_dir + 'PF' + str(sid) + '/prediction/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

mse_tmp=[]
for model in np.arange(len(models)):
    mse_tmp.append([])
    if models[model]=='linear' or models[model]=='pls' or models[model]=='ukf':
        mse_dir = data_dir + 'PF' + str(sid) + '/prediction/'+models[model]+'/'
    else:
        mse_dir = data_dir + 'PF' + str(sid) + '/prediction/'+models[model]+'_'+input+'/'
    file_prefix = 'mse_loss_32trials'
    file_surfix = 'npy'
    mse_tmp[model]=np.load(mse_dir+file_prefix+'.'+file_surfix)

# analysis the mse separately for 4 movement
# mse[movement][model] will output 8 mse number for i-th movement decoded by 6 models
testNum=8
mse=[]
for movement in range(movements):
    mse.append([])
    for model in np.arange(len(models)):
        mse[movement].append(mse_tmp[model][movement*testNum:(movement+1)*testNum])

mse_mean=[]
for movement in range(movements):
    mse_mean.append([])
    for model in np.arange(len(models)):
        mse_mean[movement].append(np.mean(mse[movement][model]))

mse_err=[]
for movement in range(movements):
    mse_err.append([])
    for model in np.arange(len(models)):
        mse_err[movement].append(np.std(mse[movement][model]))

fig, ax = plt.subplots()
x=[1,15,30,45]
color=['gold', 'orange', 'greenyellow', 'violet', 'aqua', 'hotpink']
bar_width=1
bar_mean=[]
bar_err=[]
for model in range(len(models)):
    bar_mean.append([])
    bar_err.append([])
    for movement in range(movements):
        bar_mean[model].append(mse_mean[movement][model])
    for movement in range(movements):
        bar_err[model].append(mse_err[movement][model])
for model in range(len(models)):
    ax.bar(x, bar_mean[model], yerr=bar_err[model], width=bar_width, color=color[model],
           error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2))
    x = [i + bar_width+0.5 for i in x]
ax.legend(['linear','pls','ukf','CNN+RNN','deepConv','shallowConv'],
                  loc="lower left", bbox_to_anchor=(0.758, 0.695),fontsize='small')

ax.set_xticks(x)
ax.set_xticklabels(['movement1','movement2','movement3','movement4'],rotation = 0, ha="right", position=(0,-0.04))

figname=plot_dir+'mse_compare'+str(sid)+'.pdf'
print("Plot to "+figname)
fig.savefig(figname, dpi=400)







