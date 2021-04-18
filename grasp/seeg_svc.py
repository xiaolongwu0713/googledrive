# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import GridSearchCV
import pandas as pd
import scipy.io
from grasp.config import *

pn=31
mode=1
sessions=2
motiondata={}
motions=[1,2,3,4,5]
for session in np.arange(0,2,1):
    #motions_code=list(map(lambda x:str(6)+str(x), motions))
    #motions=[61,]
    for motion in motions:
        motion_code=str(6)+str(motion)
        filename='P'+str(pn)+'_'+'H'+str(mode)+'_'+str(session+1)+'_epoch'+str(motion_code)+'ave.mat'
        file = os.path.join(processed_data,'P'+str(pn),'eeglabData',filename)
        mat=scipy.io.loadmat(file)
        data=mat["avedata"] #return np arrary. avedata is the key of this dict, data dim: eles,time,trials

        np.where(np.isnan(data).astype(int) == 1) # no nan value

        # total length of one ele is 3000 when fs=1000, now is 30 after averaged in 100point window
        # 0.5s is 0.5*fs=500 points before, now is 5 points after the average
        # extract [-0.5s,0.5s] around move onset and rearrange to |ele1_data|ele2_data|....
        ave_wind=100
        fs=1000
        start=int((1-0.5)*fs/ave_wind-1) # oneset is at 1s
        stop=int((1+0.5)*fs/ave_wind-1)
        datatmp=d=np.ones((data.shape[2],data.shape[0]*(stop-start)))
        for trial in range(0,data.shape[2]): # iter trials
            datatmp[trial]=data[:,start:stop,trial].reshape(1,-1)  # dim: concatenate all electrodes
        motiondata[str(session+1)+str(motion)]=pd.DataFrame(datatmp) # dim: trials*time
        #data.T.describe(include='all') # last trial not good

#motiondata.keys()
# '11' means first session, first motion
m1=pd.concat([motiondata['11'],motiondata['21']],axis=0,ignore_index=True) # label=1
m1label=np.ones((motiondata['11'].shape[0]+motiondata['21'].shape[0],1),int)
m2=pd.concat([motiondata['12'],motiondata['22']],axis=0,ignore_index=True) # label=2
m2label=np.ones((motiondata['12'].shape[0]+motiondata['22'].shape[0],1),int) +1
m3=pd.concat([motiondata['13'],motiondata['23']],axis=0,ignore_index=True) # label=3
m3label=np.ones((motiondata['13'].shape[0]+motiondata['23'].shape[0],1),int) +2
m4=pd.concat([motiondata['14'],motiondata['24']],axis=0,ignore_index=True) # label=4
m4label=np.ones((motiondata['14'].shape[0]+motiondata['24'].shape[0],1),int)+3
m5=pd.concat([motiondata['15'],motiondata['25']],axis=0,ignore_index=True) # label=5
m5label=np.ones((motiondata['15'].shape[0]+motiondata['25'].shape[0],1),int)+4
#del motiondata

# check train_test_split result_tmp, all good
#m11=np.concatenate((m1.to_numpy(),m1label),axis=1)
#m22=np.concatenate((m2.to_numpy(),m2label),axis=1)
#m33=np.concatenate((m3.to_numpy(),m3label),axis=1)
#m44=np.concatenate((m4.to_numpy(),m4label),axis=1)
#m55=np.concatenate((m5.to_numpy(),m5label),axis=1)

### put togather and save to disk
samples=pd.concat([m1,m2,m3,m4,m5],axis=0,ignore_index=True) #4oth trial not good
labels=np.concatenate((m1label,m2label,m3label,m4label,m5label),axis=0)
p31samplab=pd.concat([samples,pd.DataFrame(labels)],axis=1,ignore_index=True)
filename='P'+str(pn)+'_'+'H'+str(mode)+'_'+'2sessions.csv'
file = os.path.join(processed_data,'P'+str(pn),'python',filename)
p31samplab.to_csv(file,',',header=False, index=False)

###  find how many null values in the dataframe
nullcheck=samples.isnull().sum() # reture pandas series. Check all 730 column how many null in each column.
if nullcheck.sum() > 0: # total null value across all ele column
    index=np.where(nullcheck.to_numpy() > 0)


############### trianing
xtrain, xtest, ytrain, ytest = train_test_split(samples, labels,test_size=0.20,stratify=labels,random_state=1)
#xtrain=np.delete(X_train,730,1) # delete the last label verification column
#xtest=np.delete(X_test,730,1)

pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
c_range=np.arange(0.00005,0.0002,0.00001)
g_range=np.arange(0.0005,0.002,0.0001)
param_grid = [{'svc__C': param_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_range,
               'svc__gamma': g_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=8,
                  n_jobs=-1)
gs = gs.fit(xtrain, np.ravel(ytrain))
print(gs.best_score_)
print(gs.best_params_)
# testing on test set
clf = gs.best_estimator_
print('Test accuracy: %.3f' % clf.score(xtest, ytest))

################# learning curve
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_svc,
                               X=xtrain,
                               y=np.ravel(ytrain),
                               train_sizes=np.linspace(0.2, 1.0, 10),
                               cv=8,
                               n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.1, 1.03])
plt.tight_layout()
# plt.savefig('images/06_05.png', dpi=300)
plt.show()