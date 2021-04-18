from torch.nn import MSELoss, L1Loss
import numpy as np
from torch.optim import SGD, Adagrad
#from grasp.EEGnetmodel import *
from grasp.EEGnetmodelRawData import *
import matplotlib.pyplot as plt
from grasp.utils import read_fbanddata,plotloss,read_rawdata,preprocess

trainx, trainy, testx, testy =read_rawdata() # trainx: (114, 33293)
feature=trainx.shape[0]
T=15000 # stride T to create a batch
wind=1000
stride=500
batch_size=int(T/stride) # 30. batch_size = number of step in one period
totallen=trainx.shape[1] #38692

# training on one epoch
def train(model,plot='plot'):
    model.train()
    trainloses = []
    idx = 0
    while (idx < (totallen - wind - T)):  # 6080
        # x = np.zeros((len(batch_size), wind, 114))
        # y = np.zeros(len(batch_size),wind)
        x = np.zeros((30, feature, wind))
        y = np.zeros((30, wind))
        # format x into 3D tensor
        # print(idx)
        for bs in range(batch_size):
            x[bs, :, :] = trainx[:, idx:(idx + wind)]
            y[bs, :] = trainy[idx:idx + wind]
            idx = idx + stride
        target = y[:, -1]
        target = torch.from_numpy(target).float()
        target = torch.squeeze(target)
        x = torch.from_numpy(x).float()
        x = torch.unsqueeze(x, 1)  # torch.Size([30, 1, 114, 60])

        optimizer.zero_grad()
        pred = model(x)
        ls = criterion(torch.squeeze(pred), target)
        lose = ls.item()
        trainloses.append(lose)
        ls.backward()
        optimizer.step()

    if plot=='plot':
        if epoch % 20 == 0:
            plot_on_train(ax,target,pred)
    # print(f'epoch: {epoch:3} loss: {loss:10.8f}')
    trainloseavg = sum(trainloses) / len(trainloses)
    with open("trainlose.txt", "a") as f:
        f.write(str(trainloseavg) + "\n")
    print(f'epoch: {epoch:3} loss: {trainloseavg:10.8f}')

def evaluate(model,plot='plot'):
    model.eval()
    with torch.no_grad():
        testloses = []
        preds = []
        targets = []
        tidx = 0
        while (tidx < testx.shape[1] - wind - T):  # (tidx < 5*T): # check on testset first 5 move
            # x = np.zeros((len(batch_size), wind, 114))
            # y = np.zeros(len(batch_size),wind)
            tx = np.zeros((30, 19, wind))
            ty = np.zeros((30, wind))
            #print(tidx)
            # format x into 3D tensor
            for bs in range(batch_size):
                tx[bs, :, :] = testx[:, tidx:(tidx + wind)]  # test on training set or testing set
                ty[bs, :] = testy[tidx:(tidx + wind)]
                tidx = tidx + stride
            target = ty[:, -1]  # (30,)
            targets.append(target)
            tx = torch.from_numpy(tx).float()
            tx = torch.unsqueeze(tx, 1)  # torch.Size([30, 1, 114, 60])
            pred = model(tx)
            pred = torch.squeeze(pred)
            preds.append(pred)
            ls = criterion(torch.squeeze(pred), torch.squeeze(torch.FloatTensor(target)))
            testlose = ls.item()
            testloses.append(testlose)
        testlosesavg = sum(testloses) / len(testloses)
        with open("testlose.txt", "a") as f:
            f.write(str(testlosesavg) + "\n")
        if plot=='plot': # plot/noplot
            plot_on_test(ax,targets,preds)

# clear history lose
#with open('trainlose.txt', 'w'):pass
#with open('testlose.txt', 'w'):pass
fig, ax = plt.subplots(figsize=(6,3))
plt.ion()


model = EEGNet_experimental()
#model = torch.load('model1.pth')
epochs = 200
learning_rate = 0.0001
criterion = MSELoss()
#criterion = L1Loss()
# optimizer = SGD(model.parameters(), lr=learning_rate)  #,weight_decay=1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = Adagrad(model.parameters(), lr=learning_rate,weight_decay=1e-4)
for epoch in range(100,epochs):
    train(model,'noplot')
    if epoch % 1 == 0:
        evaluate(model,epoch,'plot')
torch.save(model, 'model1.pth')
plotloss(ax,'trainlose.txt','testlose.txt')



