import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
from torch.nn import MSELoss, L1Loss
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.optim import SGD, Adagrad
from grasp.EEGnetmodelRawData2 import *
import matplotlib.pyplot as plt
from grasp.utils import read_fbanddata,plotloss
from grasp.utils import read_rawdata,preprocess, rawDataSmaller
import grasp.utils as utils # enable reload the function in module without calling reload()

# training on one epoch
def train(model,dataiter,plot='plot'):
    model.train()
    trainloses = []
    for i in range(trainnum): # loop through 20 trials
        print('Trial '+ str(i+1))
        x,target=next(dataiter) #1 trial 3D x containing history force: (20, 14500, 500)
        preds=[]
        subtriallen=1450
        cycles=14500/subtriallen
        for j in range(cycles): # 14500 divided into 100 smaller batches, or you can choose a small cycle like 2 to debug
            print('SubTrial ' + str(j+1))
            xseg=x[:,j*subtriallen:(j*subtriallen+subtriallen)] #([20, 145, 500]), 145 training points
            yseg=target[j*subtriallen:(j*subtriallen+subtriallen)]

            yseg = torch.from_numpy(yseg).float()
            xseg = torch.from_numpy(xseg).float()

            xseg= xseg.permute(1, 0, 2)
            xseg = torch.unsqueeze(xseg, 1)  # ([145, 1, 20, 500]) (batch_size, channels, hight, width) including force

            optimizer.zero_grad()
            pred = model(xseg)
            preds.append(pred.detach().numpy().tolist())
            ls = criterion(torch.squeeze(pred), yseg)
            lose = ls.item()
            trainloses.append(lose)
            ls.backward()
            optimizer.step()

        if plot=='plot':
            #print(type(target))
            #print(type(preds))
            plt.cla()
            #flat_t = [item for sublist in targets for item in sublist]
            preds = [item for sublist in preds for item in sublist]
            ax.plot(target.tolist(), color="orange")
            ax.plot(preds, 'g-', lw=3)
            #plt.show()
            #plt.pause(0.2)  # Note this correction
            fig.savefig('traingPlot2.png')
            plt.close(fig)
        with open("trainlose2.txt", "a") as f:
            f.write(str(ls) + "\n")
    trainloseavg = sum(trainloses) / len(trainloses)
    print(f'epoch: {epoch:3} loss: {trainloseavg:10.8f}')

def evaluate(model,testiter,plot='plot'):
    model.eval()
    with torch.no_grad():
        testloses = []
        preds = []
        targets = []
        tidx = 0
        for i in range(testnum):  # loop through 2 trials
            print('Evaluate trial ' + str(i + 1))
            x, target = next(testiter)  # 1 trial 3D x containing history force: (20, 14500, 500)
            targets=np.concatenate((targets,target)) # save all target force
            lenth=x.shape[1]
            initialForce=np.ones((1,500))*0.1
            initialForce=np.reshape(initialForce,(1,500))
            x=x[:-1,:,:] #(19, 14500, 500)
            lengthToPredict=lenth-14000
            #x=np.concatenate((x,initialForce),axis=2)
            for j in range(lengthToPredict): # loop through all 14500 times
                xseg = x[:,j,:] #(19, 500)
                xseg=np.concatenate((xseg,initialForce),axis=0) #(20, 500)
                xseg = torch.from_numpy(xseg).float()
                xseg = torch.unsqueeze(xseg, 0)
                xseg = torch.unsqueeze(xseg,0)  # ([1, 1, 20, 500]) (batch_size, channels, hight, width)

                pred = model(xseg)

                #pred = torch.squeeze(pred)
                preds.append(pred.detach().numpy().tolist())

                # feed the current prediction
                tmpforce=list(xseg[0,0,19,:].numpy())
                tmpforce.append(pred.detach().numpy().tolist())
                tmpforce=tmpforce[1:] #500 elements
                initialForce = np.reshape(tmpforce, (1,500))

                ls = criterion(pred, torch.tensor(target[j]))
                ls = ls.item()
                testloses.append(ls)

                if j % 145 ==0 & j >0:
                    lose=sum(testloses[j-145:j])/145
                    #with open("/content/drive/MyDrive/BCI/data2/testlose2.txt", "a") as f:
                    #    f.write(str(lose) + "\n")
        if plot=='plot': # plot/noplot
            plt.cla()
            flat_t = targets[:lengthToPredict]
            flat_p=preds
            # may need to flatten list of lists
            # flat_t = [item for sublist in targets for item in sublist]
            #flat_p = [item for sublist in preds for item in sublist]
            ax.plot(flat_t, color="orange")
            ax.plot(flat_p, 'g-', lw=3)
            plt.show()
            plt.pause(0.2)  # Note this correction
            fig.savefig('evaluatePlot'+str(epoch)+'.png')

# clear history lose
#with open('trainlose.txt', 'w'):pass
#with open('testlose.txt', 'w'):pass
fig, ax = plt.subplots(figsize=(6,3))
plt.ion()
trainOrTest=1 # train=1, test=0

trainTrialnum=1
testnum=1
#trainiter=utils.gen3DOnTheFly(traindata) # generate 3D data of one trial
#testiter=utils.gen3DOnTheFly(testdata) # randomly selet 2 trial


#model = torch.load('model2.pth')
epochs = 5
learning_rate = 0.0001
criterion = MSELoss()
model = EEGNet_experimental().float()
#criterion = L1Loss()
# optimizer = SGD(model.parameters(), lr=learning_rate)  #,weight_decay=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = Adagrad(model.parameters(), lr=learning_rate,weight_decay=1e-4)
for epoch in range(epochs):
    traindata, testdata = utils.rawDataSmaller(trainTrialnum)  # randomly select 20 trials
    print('epoch :' + str(epoch))
    trainiter = utils.gen3DOnTheFly(traindata)
    train(model,trainiter,'plot')
    if epoch % 1 == 0:
        testiter = utils.gen3DOnTheFly(testdata)
        evaluate(model,testiter,'plot')
torch.save(model, 'model12.pth')
#plotloss(ax,'trainlose2.txt','testlose2.txt')



