'''
This program analysis the deep learning model learning result.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from grasp.config import *
import matplotlib as mpl
from grasp.process.utils import genSubTargetForce

mpl.rcParams['pdf.fonttype']=42

sid=1
movements=4
#optins: normalized_frequency_and_raw;frequency_and_raw;raw
input='frequency_and_raw'
M='deepConv' #shallowConv/deepConv/TSception

result_folder=root_dir+'grasp/TSception/result_'+M+'_'+input+str(sid)+'/result/'
file_prefix='prediction_epoch'
file_surfix='npy'

plot_dir = data_dir + 'PF' + str(sid) + '/prediction_of_best_trial/'+M+'_'+input+'/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

results=[]
much_more_file_index=np.arange(0,10000,2)
file_index=0
while os.path.exists(result_folder+file_prefix+str(much_more_file_index[file_index])+'.'+file_surfix):
    a_file=np.load(result_folder+file_prefix+str(much_more_file_index[file_index])+'.'+file_surfix)
    results.append(a_file)
    file_index+=1

criterion = nn.MSELoss()
losses=[]
for result in results:
    a_loss=criterion(torch.from_numpy(result[:,0]),torch.from_numpy(result[:,1]))
    losses.append(a_loss.numpy())
best_index=losses.index(min(losses))
real_best_epoch_index=best_index*2
best_result=results[best_index]

print('Save best predictions together.')
filename=plot_dir+'all_movements_best_predictions'
np.save(filename,best_result)

testNum=8
prediction=best_result[:,0]
target=best_result[:,1]
min_loss=losses[best_index]
#Note: nn.MSELoss will do mean itself
#avg_loss=min_loss/(movements*trials_per_movement)
#new_frequency=prediction.shape[0]/(trials_per_movement*movements)/15
#norm_loss=avg_loss*(default_frequency/new_frequency)

fig,ax=plt.subplots()
ax.plot(prediction,label='Predictions')
ax.plot(target,label='Ground truth')
plt.legend()
ax.set_xticks([100,350,550,800])
xlabel=['movement'+str(i) for i in np.arange(movements)]
ax.set_xticklabels(xlabel, fontsize=8)

ax.text(0.45,0.9,'MSE: '+str(np.round(min_loss, 4)),fontsize=15,transform=fig.transFigure)
#plt.pause(.02)
figname = plot_dir + 'all_movements_best_predictions.pdf'
fig.savefig(figname, dpi=400)
ax.clear()
# individual movement analysis
# moves_result[0][0]: movement 0 the first 8 trials

# collect target+prediction for each movement.
moves_best_result=[]
moves_min_loss=[]
# 8*4=32
length=int(results[0].shape[0]/32) # single trial length:28
for movement in range(movements):
    moves_best_result.append([])
    moves_min_loss.append([])
    moves_results = []
    moves_loss = []
    #for result in results: #(896, 2)
    if movement==0:
        for i in range(testNum):
            moves_results.append(best_result[i*length:(i+1)*length,:])
            a_loss = criterion(torch.from_numpy(moves_results[i][:, 0]), torch.from_numpy(moves_results[i][:, 1])).numpy()
            moves_loss.append(a_loss)
        moves_min_loss[movement]=min(moves_loss)
        moves_best_result[movement]=moves_results[moves_loss.index(min(moves_loss))]
    elif movement==1:
        for i in range(testNum):
            moves_results.append(best_result[(i+8)*length:(i+8+1)*length, :])
            a_loss = criterion(torch.from_numpy(moves_results[i][:, 0]),torch.from_numpy(moves_results[i][:, 1])).numpy()
            moves_loss.append(a_loss)
        moves_min_loss[movement] = min(moves_loss)
        moves_best_result[movement] = moves_results[moves_loss.index(min(moves_loss))]
    elif movement==2:
        for i in range(testNum):
            moves_results.append(best_result[(i+2*8)*length:(i+2*8+1)*length, :])
            a_loss = criterion(torch.from_numpy(moves_results[i][:, 0]),torch.from_numpy(moves_results[i][:, 1])).numpy()
            moves_loss.append(a_loss)
        moves_min_loss[movement] = min(moves_loss)
        moves_best_result[movement] = moves_results[moves_loss.index(min(moves_loss))]
    elif movement==3:
        for i in range(testNum):
            moves_results.append(best_result[(i+3*8)*length:(i+3*8+1)*length, :])
            a_loss = criterion(torch.from_numpy(moves_results[i][:, 0]),torch.from_numpy(moves_results[i][:, 1])).numpy()
            moves_loss.append(a_loss)
        moves_min_loss[movement] = min(moves_loss)
        moves_best_result[movement] = moves_results[moves_loss.index(min(moves_loss))]


best_individual=np.concatenate(moves_best_result,axis=0)
filename=plot_dir+'movement_individual_predictions'
np.save(filename,best_individual)

# truth force lasts 13s with 2s as start window
# generate the ground truth force
fs=1000
targetForces=[]
T=1000
step=500
for i in range(movements):
    tmp1=genSubTargetForce(i+1,fs)
    tmp2=[]
    # slide tmp like during the training
    t = 0
    while tmp1.shape[1]-1 >= t*step+T:
        tmp2.append(tmp1[0,t*step+T])
        t=t+1
    targetForces.append(tmp2)  # (648081,)
targetForces=np.concatenate(targetForces)

ax.plot(best_individual[:,0], label='Predictions')
ax.plot(best_individual[:,1],label='Force applied')
ax.plot(targetForces,label='Paradigm')
plt.legend()
ax.set_xticks([i*length+length/2 for i in np.arange(movements)])
xlabel=['task'+str(i) for i in np.arange(movements)]
ax.set_xticklabels(xlabel, fontsize=8)
min_loss=criterion(torch.from_numpy(best_individual[:,0]),torch.from_numpy(best_individual[:,1])).numpy()
#avg_loss=min_loss/movements # loss for a single trial prediction
#norm_loss=avg_loss*(default_frequency/new_frequency)
ax.text(0.45,0.9,'MSE: '+str(np.round(min_loss, 4)),fontsize=15,transform=fig.transFigure)
#plt.pause(.02) #uncomment to repeat the plot during running
print('Save best predictions individually.')
figname=plot_dir+'movement_individual_predictions.pdf'
fig.savefig(figname)

mse=[]
T=int(prediction.shape[0]/(testNum*4)) # length of one trial
for i in np.arange(testNum*4):
    mse.append([])
    mse[i]=criterion(torch.from_numpy(np.asarray(prediction[i*T:(i+1)*T]).copy()),torch.from_numpy(np.asarray(target[i*T:(i+1)*T]).copy()))\
        .cpu().detach().item()
filename= plot_dir + 'mse_loss_'+str(testNum*4)+'trials'
np.save(filename,mse)



