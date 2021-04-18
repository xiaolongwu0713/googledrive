from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adagrad
from torch.utils.data import Dataset, DataLoader
from grasp.EEGnetmodelRawData import *
from grasp.regWithSeegToTargetForce_134Truncated.utils import evaluate, train, dataset
from grasp.utils import read_fbanddata,plotloss,read_rawdata,read_rawdata_3D


basedir='/Users/long/BCI/python_scripts/grasp/regWithSeegToTargetForce_134Truncated/'
resultdir=basedir+'result/'

#99 train trails, 5 test trials
trainx, trainy, testx, testy =read_rawdata_3D('truncate','target') # trainx: (114, 33293)

#np.save('grasp/uploadToGoogleAutoML/trainx134Truncated',trainx)
#np.save('grasp/uploadToGoogleAutoML/trainy134Truncated',trainy)
#np.save('grasp/uploadToGoogleAutoML/testx134Truncated',testx)
#np.save('grasp/uploadToGoogleAutoML/testy134Truncated',testy)

#trainx=np.load(basedir+'trainx134Truncated.npy') #(19, 15000, 121)
#trainy=np.load(basedir+'trainy134Truncated.npy') #(15000, 121)
#testx=np.load(basedir+'testx134Truncated.npy') #(19, 15000, 20)
#testy=np.load(basedir+'testy134Truncated.npy') #(15000, 20)
# (19, 10001, 99)  (10001, 99)  (19, 10001, 5)  (10001, 5)
trainx=trainx.swapaxes(1,2)
trainx=trainx.swapaxes(0,1) #(121, 19, 15000)
trainy=trainy.swapaxes(0,1) #(121, 15000)
testx=testx.swapaxes(1,2)
testx=testx.swapaxes(0,1) #(20, 19, 15000)
testy=testy.swapaxes(0,1) #(20, 15000)



#train_data = trainData(torch.FloatTensor(trainx),torch.FloatTensor(trainy))

train_data = dataset(trainx,trainy)
test_data = dataset(testx,testy)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

wind=1000
stride=500
channlNum=trainx.shape[1]
kernLength1=wind//2
kernLength2=wind//2

#model = torch.load('model1.pth')
epochs = 2000
learning_rate = 0.0001
#model=torch.load(resultdir+'model_250.pth')
model = EEGNet_experimental(channlNum,kernLength1,kernLength2)
#criterion = MSELoss().float()
criterion = MSELoss()
# Adam can't even descend the training error, but hold still !!! Don't understand this.
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-4)
optimizer = Adagrad(model.parameters(), lr=learning_rate,weight_decay=1e-4)
for epoch in range(0,epochs):
    #train(model,train_loader,'plot')
    train(epoch,model,train_loader,optimizer,criterion,wind,stride,resultdir, plot='plot')
    if epoch % 10 == 0:
        #_ ,_ = evaluate(model,test_loader,'plot')
        _,_ = evaluate(epoch,model,test_loader,criterion,wind,stride,resultdir, plot='plot')
        modelfile= resultdir+ "model_%d.pth" % epoch
        torch.save(model, modelfile)
