import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)


def train(epoch,model,dataloader,optimizer,criterion,wind, stride,resultdir,plot='plot'):
    model.train()
    trainloses = []
    for i, (trainx, trainy) in enumerate(dataloader):
        trainx=torch.squeeze(trainx)  # torch.Size([19, 15000])
        trainy=torch.squeeze(trainy) # torch.Size([15000])
        idx=0
        T=trainx.shape[1] # 15000
        # x.shape: torch.Size([19, 27, 1000])
        # y.shape: torch.Size([27, 1000])
        while idx<T-wind-stride:
            if idx==0:
                x=trainx[:,idx:wind] #torch.Size([19, 1000])
                x=torch.unsqueeze(x,1) #torch.Size([19, 1, 1000])
                y = trainy[idx:wind]
                y=torch.unsqueeze(y,0) #torch.Size([1, 1000])
                idx=idx+stride #500
            else:
                tmp=trainx[:,idx:idx+wind] #torch.Size([19, 1000])
                tmp=torch.unsqueeze(tmp,1) #torch.Size([19, 1, 1000])
                x=torch.cat((x,tmp),1)
                tmp2=trainy[idx:idx+wind]
                tmp2 = torch.unsqueeze(tmp2, 0)
                y=torch.cat((y,tmp2),0)
                idx=idx+stride
        target=y[:,-1] # target is the last value
        batch_size=x.shape[1]
        x=x.permute(1,0,2)
        x= torch.unsqueeze(x,1) #torch.Size([27, 1, 19, 1000])

        optimizer.zero_grad()
        pred = model(x.float())
        ls = criterion(torch.squeeze(pred), target.float())
        lose = ls.item()
        trainloses.append(lose)
        ls.backward()
        optimizer.step()
        with open(resultdir+"trainlose.txt", "a") as f:
            f.write(str(lose) + "\n")
    if plot=='plot':
        if epoch % 10 == 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.ion()
            ax.clear()
            #flat_t = [item for sublist in target for item in sublist]
            #flat_p = [item for sublist in pred for item in sublist]
            ax.plot(target, color="orange")
            ax.plot(pred.detach().numpy(), 'g-')
            #plt.show()
            #plt.pause(0.2)  # Note this correction
            figname = resultdir+"train_epoch_%d" % epoch
            fig.savefig(figname)
            plt.close(fig)
    # print(f'epoch: {epoch:3} loss: {loss:10.8f}')
    trainloseavg = sum(trainloses) / len(trainloses)

    print(f'epoch: {epoch:3} loss: {trainloseavg:10.8f}')

def evaluate(epoch,model,dataloader,criterion,wind,stride,resultdir,plot='plot'):
    model.eval()
    with torch.no_grad():
        testloses = []
        preds = []
        targets = []
        for i, (testx, testy) in enumerate(dataloader):
            testx = torch.squeeze(testx)  # torch.Size([19, 15000])
            testy = torch.squeeze(testy)  # torch.Size([15000])
            idx = 0
            T = testx.shape[1]  # 15000
            # x.shape: torch.Size([19, 27, 1000])
            # y.shape: torch.Size([27, 1000])
            while idx < T - wind - stride:
                if idx == 0:
                    x = testx[:, idx:wind]  # torch.Size([19, 1000])
                    x = torch.unsqueeze(x, 1)  # torch.Size([19, 1, 1000])
                    y = testy[idx:wind]
                    y = torch.unsqueeze(y, 0)  # torch.Size([1, 1000])
                    idx = idx + stride  # 500
                else:
                    tmp = testx[:, idx:idx + wind]  # torch.Size([19, 1000])
                    tmp = torch.unsqueeze(tmp, 1)  # torch.Size([19, 1, 1000])
                    x = torch.cat((x, tmp), 1)
                    tmp2 = testy[idx:idx + wind]
                    tmp2 = torch.unsqueeze(tmp2, 0)
                    y = torch.cat((y, tmp2), 0)
                    idx = idx + stride
            target = y[:, -1]
            targets.append(target)
            batch_size = x.shape[1]
            x = x.permute(1, 0, 2)
            x = torch.unsqueeze(x, 1)  # torch.Size([27, 1, 19, 1000])

            pred = model(x.float())
            pred = torch.squeeze(pred).float()
            preds.append(pred)
            ls = criterion(torch.squeeze(pred), target)
            testlose = ls.item()
            testloses.append(testlose)
        testlosesavg = sum(testloses) / len(testloses)
        with open(resultdir+"testlose.txt", "a") as f:
            f.write(str(testlosesavg) + "\n")

        flat_predict = [item for sublist in preds for item in sublist]
        flat_target = [item for sublist in targets for item in sublist]
        if plot=='plot': # plot/noplot
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.ion()
            ax.clear()
            ax.plot(flat_target, color="orange")
            ax.plot(flat_predict, 'g-', lw=3)
            #plt.show()
            #plt.pause(0.2)  # Note this correction
            figname = resultdir+"test_epoch_%d" % epoch
            fig.savefig(figname)
            plt.close(fig)
        return flat_predict, flat_target
