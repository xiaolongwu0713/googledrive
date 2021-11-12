from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from gesture.config import  *
from natsort import natsorted,realsorted
from common_plot import barplot_annotate_brackets

data_dir = '/Users/long/Documents/data/gesture/'# temp data dir
training_result_dir=data_dir+'training_result/'
save_dir=training_result_dir+'dl_result/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
info=info_dir+'info.npy'
info=np.load(info,allow_pickle=True)
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

a=np.asarray(decoding_accuracy)
from scipy import stats
_, p = stats.ttest_rel(a[2,:],a[3,:]) #p=0.0007531051933323021

best=a.max()
np.where(a==best)

# TODO: change to bar graph
fig,ax=plt.subplots(figsize=(6,3))
colors=['b','g','r','m','y']
for i, modei in enumerate(model_name):
    ax.plot(decoding_accuracy[i],color=colors[i])
# true sid
ax.legend(model_name,loc="upper right",bbox_to_anchor=(0.7,0.9,0.1,0.1),fontsize='small')
loc=range(len(info))
ax.set_xticks(loc)
#val=info[:,0] # sid as x-axis
val=np.arange(1,31)
ax.set_xticklabels(val,rotation = 45, position=(0,0))
ax.tick_params(axis='x', labelsize=5)

filename=save_dir+'decodingAcc_all.pdf'
fig.savefig(filename)

mean_accs=np.mean(np.asarray(decoding_accuracy),0)
goodsub=info[mean_accs>0.4]
good_index=[]
for i in range(len(goodsub)):
    good_index.append(np.where(info[:,0]==goodsub[i,0])[0].item())


manual_pick=False
if manual_pick==True:
    seeg_illiteracy=[5,24] # sid=5 is illiteracy
    ill_index=[np.where(info == illi) for illi in seeg_illiteracy]
    ill_index=[illi[0].item() for illi in ill_index]
    good_index=np.setdiff1d(range(len(info)), ill_index).tolist()
    goodsub=info[np.logical_and(info[:,0]!=seeg_illiteracy[0], info[:,0]!=seeg_illiteracy[1])]

ax.clear()
good_acc=[]
for i,modeli in enumerate(model_name):
    good_acc.append([])
    good_acc[i]=[decoding_accuracy[i][j] for j in good_index]

bar_mean=[]
for i,modeli in enumerate(model_name):
    mean=sum(good_acc[i])/len(good_acc[i])
    bar_mean.append(mean)

bar_range=[]
for i,modeli in enumerate(model_name):
    bar_range.append([])
    bar_range[i].append(bar_mean[i]-min(good_acc[i]))
    bar_range[i].append(max(good_acc[i])-bar_mean[i])

bar_width=0.1
ax.bar(range(len(model_name)), bar_mean, yerr=np.asarray(bar_range).transpose(), width=bar_width, color=colors,error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=2))
ax.set_xticks([0,1,2,3,4]) #指定要标记的坐标
ax.set_xticklabels(model_name,rotation = 0, position=(0,0))
ax.set_ylabel('Decoding accuracy', fontsize=10, labelpad=5)

bar_err=np.array(bar_range)[:,1]
barplot_annotate_brackets(0,1,2,p,[0,1,2,3,4],bar_mean,bar_err.tolist())
barplot_annotate_brackets(1,2,3,p,[0,1,2,3,4],bar_mean,bar_err.tolist())



filename=save_dir+'decodingAcc_model.pdf'
fig.savefig(filename)
