from gesture.config import *
import torch
from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

sid=10
model_name='deepnet'
net = deepnet(208,5,500)

model_result_dir=data_dir+'training_result/model_pth/'+str(sid)+'/checkpoint_deepnet_43.pth'
checkpoint = torch.load(model_result_dir,map_location=torch.device('cpu'))

net.load_state_dict(checkpoint['net'])
params=list(net.named_parameters())
kernels=params[0][1].squeeze()
kernel=kernels[0,:].detach().numpy()

mat=np.zeros((kernels.shape[0],25))
N=len(kernel)
for ch,kernel in enumerate(kernels.detach().numpy()):
    yf = fft(kernel)
    xf = fftfreq(N, 1/1000)[:N//2]
    yf=2.0 / N * np.abs(yf[0:N // 2])
    mat[ch,:]=yf

fig,ax=plt.subplots()
vmin=0
im=ax.imshow(mat,origin='lower',cmap='RdBu_r')
filename=data_dir+'training_result/dl_result/filterOf'+model_name+'.pdf'
fig.savefig(filename)



