#%cd /content/drive/MyDrive/
# raw_data is imported from global config

#%%capture
#! pip install mne==0.19.2;
#! pip install torch;

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from grasp.TSception.utils import regulization
from grasp.utils import SEEGDataset, load_data, SEEGDataset3D, cuda_or_cup, set_random_seeds
from grasp.TSception.Models import TSception2,wholenet
from grasp.braindecode.Models import shallowConv,deepConv
from grasp.process.channel_settings import badtrials
from grasp.config import root_dir

device=cuda_or_cup()
enable_cuda = torch.cuda.is_available()
print('GPU computing: ', enable_cuda)
seed = 123456789  # random seed to make results reproducible
# Set random seed to be able to reproduce results
set_random_seeds(seed=seed)

sid=6
optins=['normalized_frequency_and_raw','frequency_and_raw','raw']
input=optins[1]

#result_dir=root_dir+'grasp/TSception/shallowConv'+str(sid)+'/'
result_dir=root_dir+'grasp/TSception/result_selection'+'_'+input+str(sid)+'/'
import os
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


#%%capture
# suppress the output
traindata, valdata, testdata = load_data(sid,split=True,move2=True,input=input)
traindata = traindata.transpose(2, 0, 1)  #-->(trials94,channels,  time)
valdata = valdata.transpose(2, 0, 1) # 32
testdata = testdata.transpose(2, 0, 1)  # 8

# Total trial number from train, val and test dataset should be equal to total trial from config file.
total_trials1=traindata.shape[0]+valdata.shape[0]+testdata.shape[0]
total_trials2=4*40-(len(badtrials[sid][0])+len(badtrials[sid][1])+len(badtrials[sid][2])+len(badtrials[sid][3]))
if total_trials1!=total_trials2:
    raise SystemExit("Trial number dones't match")
trainx, trainy = traindata[:, :-2, :], traindata[:, -2, :] #-2 is real force, -1 is target
valx, valy = valdata[:, :-2, :], valdata[:, -2, :]
testx, testy = testdata[:, :-2, :], testdata[:, -2, :]

if input=='normalized_frequency_and_raw':
    fs=250
    step=125
    T=250
else:
    fs=1000
    step=500 #ms
    T=1000 #ms
dataset_train = SEEGDataset3D(trainx, trainy,T,step)
dataset_val = SEEGDataset3D(valx, valy,T,step)
dataset_test = SEEGDataset3D(testx, testy,T,step)
train_loader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, pin_memory=False)
val_loader = DataLoader(dataset=dataset_val, batch_size=1, pin_memory=False)
test_loader = DataLoader(dataset=dataset_test, batch_size=1, pin_memory=False)

epochs=800

# TSception parameter
sampling_rate=1000
learning_rate=0.001
totalLen=trainx.shape[2] #ms
batch_size=int((totalLen-T)/step) # 280
num_T = 3 # (6 conv2d layers) * ( 3 kernel each layer)
num_S = 3
hidden_size=222
dropout=0.5
#Lambda = 1e-10

# braindecode parameter
checkshape=torch.squeeze(next(iter(test_loader))[0]) # torch.Size([28, 102, 1000])
length=checkshape.shape[2] # torch.Size([28, 102, 1000])
convfeature=40
tkernelSize=200
avgpoolKernel=100
maxpoolKernel=3
maxpoolStride=3
blockKernelSize=10
input_dim=checkshape.shape[1]

# selection neuron
M=10

#Create schedule for temperature and regularization threshold
#Create a vector of length epochs, decaying start_value to end_value exponentially, reaching end_value at end_epoch

start_temp=10
end_temp=0.1
start_thresh=10.0
end_thresh=9

def exponential_decay_schedule(start_value,end_value,epochs,end_epoch):
    t = torch.FloatTensor(torch.arange(0.0,epochs))
    p = torch.clamp(t/end_epoch,0,1)
    out = start_value*torch.pow(end_value/start_value,p)
    return out

temperature_schedule = exponential_decay_schedule(start_temp,end_temp,epochs,int(epochs*3/4))
thresh_schedule = exponential_decay_schedule(start_thresh,end_thresh,epochs,epochs)


#Network loss function
def loss_function(output,target,model,lamba,weight_decay):
    l = nn.MSELoss()
    sup_loss = l(output,target)
    reg = model.regularizer(lamba,weight_decay)
    return sup_loss,reg

weight_decay=0 #1e-6 # model parameter weight decay
lamba=0.01 # regularization of penalization of duplication selection
# def __init__(self, input_dim, M ,sampling_rate, chnNum, num_T, num_S,dropout):
net=wholenet(input_dim, M, sampling_rate,M, num_T, num_S,dropout)
#net = TSception2(sampling_rate,chnNum, num_T, num_S,dropout).float()
net.set_freeze(False)

if(enable_cuda):
    net.cuda()
#optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate,weight_decay=1e-4)


#checkpoint = torch.load(result_dir+'checkpoint440.pth')
#net.load_state_dict(checkpoint['net'])
#optimizer.load_state_dict(checkpoint['optimizer'])

fig2, ax2=plt.subplots()

debugg = False
#debugg=True
for epoch in range(epochs):
    print("------ epoch " + str(epoch) + " -----")
    net.train()
    net.set_thresh(thresh_schedule[epoch])
    net.set_temperature(temperature_schedule[epoch])

    loss_epoch = 0
    # trial=0
    for trial, (trainx, trainy) in enumerate(train_loader):  # ([1, 15000, 19]), ([1, 15000])
        if debugg == True:  # just test one trial
            if trial == 1:
                break
                pass
        optimizer.zero_grad()

        if (enable_cuda):
            x = trainx.float().cuda()
            target = trainy.float().cuda()
        else:
            x = trainx.float()
            target = trainy.float()
        y_pred = net(x)
        # target = torch.from_numpy(target)

        sup_loss, reg = loss_function(y_pred, target.float(), net,lamba, weight_decay)
        loss = sup_loss + reg  # regulization
        loss_epoch+=sup_loss.item()

        loss.backward()
        optimizer.step()

    #H, sel, probas = net.monitor()
    #ax2.plot(probas.detach().cpu().numpy())
    #fig2.savefig(result_dir + 'prob_dist' + str(epoch) + '.png')
    #ax2.clear()

    print("" + str(epoch) + " loss:" + str(loss_epoch / (trial + 1)) + ".")
    if epoch % 2 == 0:
        net.eval()
        print("Validating...")
        with torch.no_grad():
            vpredAll = []
            vtargetAll = []
            for trial, (vx, vtarget) in enumerate(val_loader):  # ([1, 15000, 19]), ([1, 15000])
                if (enable_cuda):
                    vx = vx.float().cuda()
                    vtarget = vtarget.float().cuda()
                else:
                    vx = vx.float()
                    vtarget = vtarget.float()
                y_pred = net(vx)
                

                y_pred = y_pred.squeeze().cpu().detach().numpy()
                vtarget = vtarget.squeeze().cpu().numpy()
                vpredAll.append(y_pred)
                vtargetAll.append(vtarget)

        vpredAll = np.concatenate(vpredAll, axis=0)
        vtargetAll = np.concatenate(vtargetAll, axis=0)
        #loss_val = criterion(torch.from_numpy(vpredAll.squeeze()), torch.from_numpy(vtargetAll.squeeze()))

        pred_target=np.concatenate((vpredAll[:,None],vtargetAll[:,None]),axis=1)
        save_pred=result_dir + 'prediction_epoch' + str(epoch) + '.npy'
        np.save(save_pred, pred_target)

        fig, ax = plt.subplots(figsize=(6, 3))
        plt.ion()
        ax.clear()
        ax.plot(vtargetAll, label="True", linewidth=1)
        ax.plot(vpredAll, label='Predicted - Test', linewidth=1)
        ax.legend(loc='upper left')
        figname = result_dir + 'prediction' + str(epoch) + '.png'
        fig.savefig(figname)
        plt.close(fig)
    if epoch % 10 == 0:
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        savepath = result_dir + 'checkpoint' + str(epoch) + '.pth'
        torch.save(state, savepath)
