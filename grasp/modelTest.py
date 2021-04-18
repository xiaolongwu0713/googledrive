import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from grasp.regWithSeegToTargetForce_134Truncated.utils import dataset, evaluate
from grasp.utils import readRestForModelTest, read_rawdata_3D

basedir='/Users/long/BCI/python_scripts/grasp/regWithSeegToTargetForce_134Truncated/'
resultdir=basedir+'modelTest/'

#trainx, trainy, testx, testy =read_rawdata_3D()
testx,testy=readRestForModelTest()
testx=testx.swapaxes(1,2)
testx=testx.swapaxes(0,1) #(20, 19, 15000)
testy=testy.swapaxes(0,1) #(20, 15000)

test_data = dataset(testx,testy)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)


wind=1000
stride=500
criterion = MSELoss()
model=torch.load(basedir+'result_tmp/model_290.pth')
pred,target=evaluate(1,model,test_loader,criterion,wind,stride,resultdir, plot='noplot')
ls = criterion(torch.squeeze(torch.Tensor(pred)), torch.Tensor(target))
testlose = ls.item()
print('Test lose: '+str(testlose))

fig, ax = plt.subplots(figsize=(6, 3))
plt.ion()
ax.plot(pred, 'orange', label='predicted force')
ax.plot(target, 'green', label='target force',lw=3)
plt.legend()
#plt.show()
#plt.pause(0.2)  # Note this correction
figname = resultdir+'testResult'
fig.savefig(figname)
plt.close(fig)