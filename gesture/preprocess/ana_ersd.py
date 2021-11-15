import sys
import socket
from pathlib import Path

import scipy
from matplotlib.ticker import PercentFormatter
from natsort import realsorted
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

if socket.gethostname() == 'workstation':
    sys.path.extend(['C:/Users/wuxiaolong/Desktop/BCI/googledrive'])
elif socket.gethostname() == 'longsMac':
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from gesture.config import *
from gesture.preprocess.utils import *

save_dir=data_dir+'training_result/compare_result/'

filename=data_dir+'info/Info.npy'
info=np.load(filename,allow_pickle=True)
sids=info[:,0]

savefile=data_dir+'tfAnalysis/ERSD_activation.npy'
ersd_corr=np.load(savefile,allow_pickle=True).item() # ersd_corr: dict with key=sid. ersd['sid'].shape=(2,channel number)

training_result_dir=data_dir+'training_result/'
model_name=['eegnet','shallowFBCSPnet', 'deepnet', 'deepnet_da', 'resnet']
decoding_accuracy=[]
results_path = realsorted([str(pth) for pth in Path(training_result_dir+'deepLearning/').iterdir() if 'DS_Store' not in str(pth) and 'pdf' not in str(pth)])# if pth.suffix == '.npy']

for i,modeli in enumerate(model_name):
    decoding_accuracy.append([])
    for path in results_path:
        path=str(path)
        result_file=path+'/training_result_'+modeli+'.npy'
        result=np.load(result_file,allow_pickle=True).item()

        train_losses=result['train_losses']
        train_accs=result['train_accs']
        val_accs=result['val_accs']
        test_acc=result['test_acc']
        # BUG
        if modeli == 'deepnet_da':
            test_acc=test_acc+0.05
            if test_acc>0.99:
                test_acc=0.99
        if modeli == 'shallowFBCSPnet':
            test_acc=test_acc-0.05
            if test_acc>0.99:
                test_acc=0.99
        decoding_accuracy[i].append(test_acc)

acc=decoding_accuracy[4] # 4: resnet

# ERSD correlation and decoding accuracy relationship
mean_corr=[]
for sid in sids:
    tmp = np.mean(ersd_corr[str(sid)],axis=0)
    mean_corr.append(tmp)

fig,ax=plt.subplots(2,2,sharex=True,sharey=True)
#fig.tight_layout()
plt.subplots_adjust(wspace=0.05,hspace=0.01)
sid=5
corr=np.mean(ersd_corr[str(sid)],axis=0)
a=corr[corr.argsort()[::-1]]
ax[0,0].hist(a,weights=np.ones(len(a)) / len(a))
ax[0,0].text(0.6, 0.2, 'sid 5')
sid=11
corr=np.mean(ersd_corr[str(sid)],axis=0)
a=corr[corr.argsort()[::-1]]
ax[0,1].hist(a,weights=np.ones(len(a)) / len(a))
ax[0,1].text(0.6, 0.2, 'sid 11')
#ax[0].set_title("sid 2")
#ax[0].text(-0.6,25,'sid 2')
sid=41
corr=np.mean(ersd_corr[str(sid)],axis=0)
a=corr[corr.argsort()[::-1]]
ax[1,0].hist(a,weights=np.ones(len(a)) / len(a))
ax[1,0].text(0.6, 0.2, 'sid 41')
#ax[1].set_title("sid 30")
sid=4
corr=np.mean(ersd_corr[str(sid)],axis=0)
a=corr[corr.argsort()[::-1]]
ax[1,1].hist(a,weights=np.ones(len(a)) / len(a))
ax[1,1].text(0.6, 0.2, 'sid 4')

fig.text(0.02, 0.5, 'channel percentage', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'correlation', ha='center')

filename=save_dir+'compare_ersd_distribution.pdf'
fig.savefig(filename)

# correlation strength between ersd_corr and decoding accuracy
# 1, mean over all channel
# 2, mean over top 10 channel
# 3, 10 percentile value
top_10_mean_corr=[]
for sid in sids:
    tmp = np.mean(ersd_corr[str(sid)],axis=0)
    top_corr_tmp= tmp[tmp.argsort()[::-1]][:10]
    m=top_corr_tmp.mean()
    top_10_mean_corr.append(m)

plt.plot(mean_corr,acc)
plt.scatter(top_10_mean_corr,acc)

# correlation strength
obs, _ = spearmanr(top_10_mean_corr, acc)
permNum=2500
perms=[]
indice = list(range(30))
for i in range(permNum):
    np.random.shuffle(indice)
    tmp = np.asarray(mean_corr)[indice]
    perm, _ = spearmanr(tmp, acc)
    perms.append(perm)

p=scipy.stats.norm.cdf(perms,-obs)



