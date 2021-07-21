'''
This program will compare the decoding results between raw input and frequency+raw input for all subjects.
Then a bar plot will be made.
'''

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import torch.nn as nn
from grasp.config import *
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42


#Note: sid=2 will return nan from lstm1 during to some unstable calculation. solution: normalization before deep learning.
sids=[1,2,6,10,16]
inputs=['frequency_and_raw','raw'] # frequency_and_raw;raw
models=['TSception','deepConv','shallowConv'] #'TSception'/'deepConv'/'shallowConv'/
model='TSception'
movements=4

summary_dir=data_dir+'summary'+'/'
plot_dirs=[]
for sid in range(len(sids)):
    plot_dirs.append([])
    for input in range(len(inputs)):
        plot_dirs[sid].append(data_dir + 'PF' + str(sids[sid]) + '/prediction/'+model+'_'+inputs[input]+'/')
        if not os.path.exists(plot_dirs[sid][input]):
            os.makedirs(plot_dirs[sid][input])

# result_dir of sid 1 = result_dirs[1][0] or result_dirs[1][1]
result_dirs=[]
for sid in range(len(sids)):
    result_dirs.append([])
    for input in range(len(inputs)):
        result_dirs[sid].append(root_dir+'grasp/TSception/result_'+model+'_'+inputs[input]+str(sids[sid])+'/result/')
file_prefix='prediction_epoch'
file_surfix='npy'


results_tmp=[]
results=[]
much_more_file_index=np.arange(0,10000,2)
for sid in range(len(sids)):
    results_tmp.append([])
    for input in range(len(inputs)):
        results_tmp[sid].append([])
        file_index = 0
        while os.path.exists(result_dirs[sid][input]+file_prefix+str(much_more_file_index[file_index])+'.'+file_surfix):
            a_file=np.load(result_dirs[sid][input]+file_prefix+str(much_more_file_index[file_index])+'.'+file_surfix)
            results_tmp[sid][input].append(a_file)
            file_index+=2

# normalize result and prediction.
# when using different standardization/normalization method when reading data, the loss would be different.
scaler =  MinMaxScaler(feature_range=(0,2))
T=results_tmp[0][0][0].shape[0]
for sid in range(len(sids)):
    results.append([])
    for input in range(len(inputs)):
        results[sid].append([])
        for i in range(len(results_tmp[sid][input])):
            tmp=np.reshape(results_tmp[sid][input][i].T,(1,-1))
            tmp=scaler.fit_transform(tmp.T)
            results[sid][input].append(np.asarray([[tmp[:T,0]],[tmp[T:,0]]]).squeeze().T)


criterion = nn.MSELoss()
losses=[]
real_best_epoch_index=[]
best_result=[]
best_index=[]
for sid in range(len(sids)):
    losses.append([])
    real_best_epoch_index.append([])
    best_result.append([])
    best_index.append([])
    for input in range(len(inputs)):
        losses[sid].append([])
        real_best_epoch_index[sid].append([])
        best_result[sid].append([])
        best_index[sid].append([])
        for result in results[sid][input]:
            a_loss=criterion(torch.from_numpy(result[:,0]),torch.from_numpy(result[:,1]))
            losses[sid][input].append(a_loss.numpy())
        best_index[sid][input]=losses[sid][input].index(min(losses[sid][input]))
        real_best_epoch_index[sid][input]=best_index[sid][input]*2
        best_result[sid][input]=results[sid][input][best_index[sid][input]]

        filename = plot_dirs[sid][input] + 'all_predictions'
        np.save(filename, best_result[sid][input])


print('Save best predictions of all testing trials.')
fig,ax=plt.subplots()
fig2=plt.figure()
ax1 = fig2.add_subplot(231)
ax2 = fig2.add_subplot(232)
ax3 = fig2.add_subplot(233)#, sharex=ax1, sharey=ax1)
ax4 = fig2.add_subplot(234) #, sharex=ax1, sharey=ax1)
ax5 = fig2.add_subplot(235) #, sharex=ax1, sharey=ax1)
#ax2.get_yaxis().set_ticks([])
#ax3.get_yaxis().set_ticks([])
#ax5.get_yaxis().set_ticks([])
plt.tight_layout()
ax_all=[ax1,ax2,ax3,ax4,ax5]

box4 = ax4.get_position()
box4.x0 = box4.x0+0.1
box4.x1 = box4.x1+0.1
ax4.set_position(box4)

box5 = ax5.get_position()
box5.x0 = box5.x0+0.1
box5.x1 = box5.x1+0.1
ax5.set_position(box5)


trials_per_movement=8
length=int(results[0][0][0].shape[0]/32) # single trial length:28
pred=[]
tgt=[]
best_one=[]
best_loss=[]
for sid in range(len(sids)):
    pred.append([])
    tgt.append([])
    best_one.append([])
    best_loss.append([])
    for input in range(len(inputs)):
        pred[sid].append([])
        tgt[sid].append([])
        best_one[sid].append([])
        best_loss[sid].append([])

        prediction=best_result[sid][input][:,0]
        target=best_result[sid][input][:,1]
        min_loss=losses[sid][input][best_index[sid][input]]

        # pick best individual prediction from the best epoch result
        for movement in range(movements):
            pred[sid][input].append([])
            tgt[sid][input].append([])
            best_one[sid][input].append([])
            best_loss[sid][input].append([])
            for i in range(trials_per_movement):
                pred[sid][input][movement].append(prediction[movement * trials_per_movement * length + i * length:
                                                             movement * trials_per_movement * length + (
                                                                         i + 1) * length])
                tgt[sid][input][movement].append(target[movement * trials_per_movement * length + i * length:
                                                        movement * trials_per_movement * length + (i + 1) * length])
            movement_loss=[criterion(torch.from_numpy(pred[sid][input][movement][i]),torch.from_numpy(tgt[sid][input][movement][i]))
                           for i in range(trials_per_movement)]
            best_one_index=movement_loss.index(min(movement_loss))
            best_one[sid][input][movement]=[pred[sid][input][movement][best_one_index],tgt[sid][input][movement][best_one_index]]
            best_loss[sid][input][movement]=min(movement_loss)

        # sanity check
        #color = ['green','red','yellow','black']
        #for movement in range(movements):
        #    for i in range(trials_per_movement):
        #        ax.plot(tgt[sid][input][movement][i], color=color[movement])

        # plot all testing trail
        ax.clear()
        ax.plot(prediction,label='Predictions')
        ax.plot(target,label='Ground truth')
        ax.legend()
        ax.set_xticks([100,350,550,800])
        xlabel=['movement'+str(i) for i in np.arange(movements)]
        ax.set_xticklabels(xlabel, fontsize=8)
        ax.text(0.45,0.9,'MSE: '+str(np.round(min_loss, 4)),fontsize=15,transform=fig.transFigure)
        ax.set_ylabel('Force amplitude')
        plt.pause(.02)
        figname = plot_dirs[sid][input] + 'all_movements_best_predictions.png'
        fig.savefig(figname, dpi=400)

        # plot the best individual from the best epoch
        ax.clear()
        best_pred=np.asarray(best_one[sid][input]).transpose(1,0,2)
        best_pred=np.reshape(best_pred,(2,-1))
        filename = plot_dirs[sid][input] + 'individual_predictions'
        np.save(filename, best_pred)
        ax.plot(best_pred[0, :],label='Predictions')
        ax.plot(best_pred[1, :],label='Ground truth')
        ax.legend()
        ax.set_xticks([14, 40, 60, 90])
        xlabel = ['movement' + str(i) for i in np.arange(movements)]
        ax.set_xticklabels(xlabel, fontsize=8)
        ax.text(0.45, 0.9, 'MSE: ' + str(np.round(min_loss, 4)), fontsize=15, transform=fig.transFigure)
        ax.set_ylabel('Force amplitude')
        plt.pause(.02)
        figname = plot_dirs[sid][input] + 'individual_predictions.png'
        fig.savefig(figname, dpi=400)

    # plot the loss bar of two inputs for each subject
    best_loss_tmp = np.asarray(best_loss)
    ax.clear()
    X = np.arange(4)
    color=['b','g']
    width = 0.25
    for input in range(len(inputs)):
        ax.bar(X + input*width, best_loss_tmp[sid,input,:], color=color[input], width=width, label=inputs[input])
        ax_all[sid].bar(X + input * width, best_loss_tmp[sid, input, :], color=color[input], width=width,label=inputs[input])
        ax_all[sid].set_xticks([0 + width / 2, 1 + width / 2, 2 + width / 2, 3 + width / 2])
        xlabel = ['grasp' + str(i) for i in np.arange(movements)]
        ax_all[sid].set_xticklabels(xlabel, fontsize=8, rotation=45, va="center", position=(0,-0.04))
        ax_all[sid].tick_params(axis='y', labelsize=8)
    ax_all[0].set_ylabel('MSE',fontsize=8,labelpad=2)
    ax_all[3].set_ylabel('MSE', fontsize=8, labelpad=2)

    ax.legend()
    ax.set_xticks([0+width/2, 1+width/2, 2+width/2, 3+width/2])
    xlabel = ['grasp' + str(i) for i in np.arange(movements)]
    ax.set_xticklabels(xlabel, fontsize=8)
    ax.set_ylabel('MSE')
    figname = summary_dir + 'input_compare_bar'+str(sid)+'.pdf'
    fig.savefig(figname, dpi=400)
ax_all[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop={'size':8})
print('Save to '+summary_dir)
figname = summary_dir + 'input_compare_bar_all.pdf'
fig2.savefig(figname, dpi=400)






