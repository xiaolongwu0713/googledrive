import numpy as np
from gesture.config import  *
result_dir=data_dir+'training_result/'
sid=10
model_name='resnet'
result_file=result_dir+str(sid)+'/training_result_'+model_name+'.npy'
result=np.load(result_file,allow_pickle=True).item()

train_losses=result['train_losses']
train_accs=result['train_accs']
val_accs=result['val_accs']
test_acc=result['test_acc']

assert len(train_losses)==len(train_accs)==len(val_accs)
print(len(val_accs))





