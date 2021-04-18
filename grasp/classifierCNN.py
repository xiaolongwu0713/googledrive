
from grasp.config import *
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


pn=31
mode=1

'''
### loading data, data format(each row): | ele1...|ele2..|..| --> 1/2/3/4/5
filename='P'+str(pn)+'_'+'H'+str(mode)+'_'+'2sessions.csv'
file = os.path.join(processed_data,'P'+str(pn),'python',filename)
dataset=pd.read_csv(file,',',header=None, index_col=None)
sample = dataset.iloc[:, :-1]
lbls = dataset.iloc[:,-1]
'''

data={}
samples={}
motions=[1,2,3,4,5]

for session in np.arange(0,2):
    #motions_code=list(map(lambda x:str(6)+str(x), motions))
    #motions=[61,]
    for motion in motions:
        motion_code=str(6)+str(motion)
        filename='P'+str(pn)+'_'+'H'+str(mode)+'_'+str(session+1)+'_epoch'+str(motion_code)+'ave.mat'
        file = os.path.join(processed_data,'P'+str(pn),'eeglabData',filename)
        mat=scipy.io.loadmat(file)
        dataLoaded=mat["avepower"] #return np arrary. avedata is the key of this dict, data dim: eles,time,trials

        #np.where(np.isnan(data).astype(int) == 1) # no nan value
        ave_wind=100
        fs=1000
        start=int((1-0.5)*fs/ave_wind-1) # oneset is at 1s
        stop=int((1+0.5)*fs/ave_wind-1)
        data[str(session+1)+str(motion)]=dataLoaded[:,start:stop,:]

for _, k in enumerate(data):
    cube=data[str(k)] # ele * time * trial
    mean2d=cube.mean(axis=1) # np.ndarray
    mean=np.repeat(mean2d[:,np.newaxis,:],cube.shape[1],axis=1)
    cubeMean=cube-mean
    std2d=cubeMean.std(axis=1)
    std = np.repeat(std2d[:, np.newaxis, :], cubeMean.shape[1], axis=1)
    cubeNorm=cubeMean/std # element-wise division
    samples[str(k)]=cubeNorm

samples=np.concatenate([samples["11"],samples["12"],samples["13"],samples["14"],samples["15"],
                  samples["21"],samples["22"],samples["23"],samples["24"],samples["25"]],2)
samples=np.moveaxis(samples, -1, 0) # move from ele*time*trial to trial*ele*time

target11=np.zeros(10,int)
target12=np.zeros(10,int)+1
target13=np.zeros(10,int)+2
target14=np.zeros(10,int)+3
target15=np.zeros(10,int)+4
target21=np.zeros(10,int)
target22=np.zeros(10,int)+1
target23=np.zeros(10,int)+2
target24=np.zeros(10,int)+3
target25=np.zeros(9,int)+4
target_tmp=np.concatenate([target11,target12,target13,target14,target15,target21,target22,target23,target24,target25],0)

'''
def one_hot(y):
    lbl = np.zeros(5)
    lbl[y-1] = 1
    return lbl
## target: trial*5=99*5
targets = []
for value in target_tmp:
    targets.append(one_hot(value))
targets = np.array(targets)
'''

## shuffle samples and targets
shuf=np.arange(samples.shape[0])
np.random.shuffle(shuf)
samples=samples[shuf]
targets=target_tmp[shuf]


model = Sequential()
model.add(layers.Conv2D(16, 3, strides=2,input_shape=(73,10,1), padding='same',data_format="channels_last"))
model.add(layers.ReLU())
#model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(32, 3, strides=2, padding='same',data_format="channels_last"))
model.add(layers.ReLU())
#model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(64, 3, strides=2, padding='same',data_format="channels_last"))
model.add(layers.ReLU())
#model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(5)) # 5 classes
#model.add(layers.BatchNormalization())
model.summary()

save_path = os.path.join(processed_data,'P'+str(pn),'python','model.h5')
if os.path.isfile(save_path):
    model.load_weights(save_path)
    print('reloaded.')

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## update the learning rate
def lr_scheduler(epoch):
    # half the lr every 5 epochs
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)
lrate = LearningRateScheduler(lr_scheduler)

# conv2D expect a 4-D input
samplestmp=samples.reshape(samples.shape[0],samples.shape[1],samples.shape[2],1)

epochs=10
history = model.fit(
    samplestmp,targets,
    batch_size=5,
    validation_split=0.2,
    epochs=epochs
)
#history = model.fit(sample, target, epochs=400,batch_size=128, validation_split=0.2,verbose=2, callbacks=[lrate])

model.save_weights(save_path)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()