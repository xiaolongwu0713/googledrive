from gesture.config import *
import torch
from gesture.models.deepmodel import deepnet,deepnet_seq,deepnet_rnn, deepnet_da


sid=10
model_name='deepnet'
net = deepnet(208,5,500)

model_result_dir=data_dir+'training_result/model_pth/'+str(sid)+'/checkpoint_deepnet_46.pth'
checkpoint = torch.load(model_result_dir,map_location=torch.device('cpu'))

net.load_state_dict(checkpoint['net'])

net=




