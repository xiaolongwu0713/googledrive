# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:54:35 2018

@author: prash
"""

#import scipy.sparse as sps
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression	

import scipy.io
import numpy as np
from scipy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from keras import optimizers
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from scipy.signal import butter,cheby1,lfilter
#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
#from keras import regularizers
#from keras import losses
from keras import optimizers
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
#from keras.layers.normalization import BatchNormalization
#from keras.layers.core import Dense,Dropout,Activation,Lambda
##from pyglmnet import GLM #change pwd to E:\Brain Machine\Thesis\Potential datasets\BCI Competition 2008 – Graz data set B\pyglmnet-master
#import matplotlib.pyplot as plt
import random
random.seed(1)

import warnings
warnings.filterwarnings("ignore")

def lda(feat_train,labels_train,feat_test):
#    print("\nLinear Discriminant Analysis")
    clf=LinearDiscriminantAnalysis()
    clf.fit(feat_train,labels_train)
    pred_train=clf.predict(feat_train)
    pred_test=clf.predict(feat_test)
    return pred_train,pred_test    

def neural_net(feat_train,ltrain,feat_test,neuron):
    ltrain=np.array(ltrain)
    ltrain[np.where(ltrain==1)]=0
    ltrain[np.where(ltrain==2)]=1
#    print("\nSingle layered Feedforward neural network with %d neurons"%neuron)
    model=Sequential()
    model.add(Dense(100,input_dim=np.shape(feat_train)[1],
                activation='sigmoid',kernel_initializer='uniform'))
    #model.add(Dropout(0.05))
    #model.add(BatchNormalization())
#    model.add(Dense(50,activation='sigmoid',kernel_initializer='uniform'))
    ##model.add(BatchNormalizafrom keras.layers import Densetion())
#    model.add(Dense(90,activation='sigmoid',kernel_initializer='uniform'))
#    model.add(Dense(15,activation='sigmoid',kernel_initializer='uniform'))
    model.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))
    #optim=optimizers.SGD(lr=0.0001)
    model.compile(loss='mean_squared_error',optimizer=optimizers.adam(lr=0.00001),metrics=['accuracy'])
    model.fit(feat_train,ltrain,verbose=0)

    pred_train=(model.predict(feat_train))
    
    pred_test=(model.predict(feat_test))
    return pred_train,pred_test


#train_error=np.mean(labels_train!=np.array(pred_train))
#print("train error:",train_error)
def naive_bayes(feat_train,y_train,feat_test):
    gnb=GaussianNB()
    pred_train=gnb.fit(feat_train,y_train).predict(feat_train)
    pred_test=gnb.fit(feat_train,y_train).predict(feat_test)
    return pred_train,pred_test

def KNN(feat_train,y_train,feat_test,neighbor):
#    print("\n%d-Nearest neighbor" % neighbor)
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(feat_train, y_train)
    pred_train=neigh.predict(feat_train)
    pred_test=neigh.predict(feat_test)
    return pred_train,pred_test

def log_reg(feat_train,y_train,feat_test):
#    print("\nLogistic regression")
    clf=LogisticRegression(random_state=0,solver='liblinear',multi_class='ovr').fit(feat_train,y_train)
#    clf=linear_model.Ridge(alpha=.1).fit(feat_train,y_train)
    pred_train=clf.predict(feat_train)  
    pred_test=clf.predict(feat_test)    
    return pred_train,pred_test

def boost(feat_train,y_train,feat_test,depth):
#    print("\nXGBOOST max_depth:",depth)
#    clf=xgb.XGBClassifier(max_depth,min_child_weight=1,n_estimators=1000)
#    dtrain=xgb.DMatrix(features_train,labels_train)
    clf=xgb.XGBClassifier(depth=depth,
                seed= 0, #for reproducibility
                silent= 1,
                learning_rate= 0.05,
                n_estimators= 500)
    clf.fit(feat_train,y_train,verbose=False)
    pred_train=clf.predict(feat_train)
    pred_test=clf.predict(feat_test)
    return pred_train,pred_test

#    print("\nRandom forest with max depth:",depth)
def random_forest(feat_train,labels_train,feat_test,depth):

    model=RandomForestClassifier(
          max_depth=depth,random_state=0,n_estimators=1500)
    #      min_samples_leaf=10,
    #      min_weight_fraction_leaf= 0.4,n_estimators= 5000)
    model.fit(feat_train,labels_train)
    pred_train=model.predict(feat_train)
    pred_test=model.predict(feat_test)
    return pred_train,pred_test
##########################################
#band pass butter worth filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def cheby1_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#    b, a = cheby1_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
########################################
def covariance(X):
    return np.dot(X,X.T)/np.trace(np.dot(X,X.T))

def get_feat(data,sf):
    return np.log(np.var(np.dot(sf.T,data.T),axis=1)/sum(np.var(np.dot(sf.T,data.T),axis=1))) #check axis

def get_spatial(sum_left,sum_right,J):
    C=sum_right+sum_left
    eigvals,eigvecs=LA.eig(C)
#    sort_eigvals=np.sort(eigvals)[::-1]
    diag_inv=np.zeros((C.shape[1],C.shape[1]))
    for i in range(eigvals.shape[0]):
        diag_inv[i,i]=(1/np.abs(eigvals[i].real)) #considering absolute value of the real parts! need to verify if approach is correct  

    P=np.sqrt(diag_inv)*eigvecs.T
    S_l=P*sum_left*P.T
    S_r=P*sum_right*P.T   

    E1,U1=LA.eig(S_l,S_r)
    ord1 = np.argsort(E1)
    ord1 = ord1[::-1]
    E1 = E1[ord1]
    U1 = U1[:,ord1]
    W=np.dot(U1,P)#projection matrix
#consider the first 10 columns of W as tzeroshe required feautres 
#ideally you want to pick the first three features and the last three
    W_select=np.zeros([np.shape(W)[0],2*J])
    W_select[:,0:J]=W[:,0:J]
    W_select[:,J::]=W[:,np.shape(W)[1]-J:np.shape(W)[1]]

    return W_select
#plt.plot(xgb_train_err,label="Train")
#plt.plot(xgb_test_err,label="Test")
#plt.title("XGBoost")
#plt.legend()
#plt.show()



#%%
first_col=[i for i in range(6,13)]
second_col=[i for i in range(8,15)]
freq=np.zeros([15,2])

c=14
d=19
while(d<=40):
  second_col.append(d)
  d+=3
while(c<=35):
  first_col.append(c)
  c+=3
freq[:,0]=first_col
freq[:,1]=second_col
print(freq.shape)


#%% Common Spatial Pattern
# class1=right hand, class 2= right leg
num_sub=14
features_test=dict()
features_train=dict()
features_train_tmp=dict()
model=dict()
labels_train=dict()
labels_test=dict()
num_channels=15
#sum_hand=np.zeros((num_channels,num_channels))
#sum_leg=np.zeros((num_channels,num_channels))
labels=[]
sample_rate=512
spatial_filt=dict()
num_feat=3 #times 2
#freq=[4,8,6,10,8,12,10,14,12,16,14,18,16,20,18,22,20,24,22,26,24,28,26,30]
#freq=[8,16,24,32,40] #change according to bands required 
feat_best=6
for sub in range(num_sub): # looping through subjects 
    features_train[sub]=np.zeros([100,(2*num_feat)])
    labels_train[sub]=[]
    if (sub+1)<10:    
        file_train='S0%dT.mat'% (sub+1)
        mat_train=scipy.io.loadmat(file_train)
    else:
        file_train='S%dT.mat'% (sub+1)
        mat_train=scipy.io.loadmat(file_train)
    data_train=mat_train['data']
    spatial_filt[sub]=dict()
    key_count=0
    for freq_count in range(freq.shape[0]): # loop for frequency
        tmp_feat=[]
#         lower=freq[freq_count]
#         if lower == freq[-1]:
#             break
#         higher=freq[freq_count+1]
#         if lower>higher:
#             continue
        lower=freq[freq_count,0]
        higher=freq[freq_count,1]
        sum_hand=np.zeros((num_channels,num_channels))
        sum_leg=np.zeros((num_channels,num_channels))
        hand=0
        leg=0
        for k in range(5): #loop through trials
            
            cell_train=data_train[0][k]
            X_train=cell_train[0][0][0]
            X_train_filt=bandpass_filter(X_train,lowcut=lower, highcut=higher, fs=512, order=5) #check the sampling rate !!!!
            time_train=cell_train[0][0][1][0]##remove last if prob
            if freq_count==0:
                labels_tmp_train=cell_train[0][0][2]
                labels_train[sub].extend(labels_tmp_train[0])
            var=0
            for l_tmp in range(len(labels_tmp_train[0])):
                if labels_tmp_train[0][l_tmp]==1:
    #                train_leg[sub].append
                    sum_hand+=covariance(X_train_filt[var+int(4.5*sample_rate):var+8*sample_rate,:].T) #transpose because we need num_channel vs num_channel
                    hand+=1
                else:
                    sum_leg+=covariance(X_train_filt[var+int(4.5*sample_rate):var+8*sample_rate,:].T)
                    leg+=1
                var=time_train[l_tmp] ###add [0] if prob
        mean_hand=sum_hand/hand
        mean_leg=sum_leg/leg
        spatial_filt[sub][key_count]=(get_spatial(mean_hand,mean_leg,num_feat))
        
        for k in range(5):
            cell_train=data_train[0][k]
            X_train=cell_train[0][0][0]
            X_train_filt=bandpass_filter(X_train,lowcut=lower, highcut=higher, fs=512, order=5) 
            time_train=cell_train[0][0][1][0] ##remove last if prob
            #Computing Spatial features for training
            var=0
            for count in range(len(time_train)): 
                tmp=get_feat(X_train_filt[var+int(4.5*sample_rate):var+8*sample_rate,:],np.array(spatial_filt[sub][key_count]))
#                if freq_count==0:
#                    features_train[sub]=tmp
#                else:
                tmp_feat.append(tmp)
                var=time_train[count] ##remove last if prob
                
        tmp_feat=np.array(tmp_feat)
        if freq_count==0:
            features_train[sub]=tmp_feat
        else:
            features_train[sub]=np.concatenate((features_train[sub],tmp_feat),axis=1)
#        print("In training:",key_count)
        key_count=key_count+1
    model[sub]=sklearn.feature_selection.SelectKBest(mutual_info_classif,k=feat_best).fit(features_train[sub],labels_train[sub])
    features_train[sub]=model[sub].transform(features_train[sub])
#    np.random.shuffle(features_train[sub])# to introduce randomness shuffle all the features
print("Training Features Computed\n Computing Testing features...")
#print("In traning final key_count:",key_count)
#%%
from sklearn.decomposition import PCA
for sub in range(num_sub):
    pca=PCA().fit(features_train[sub])
    plt.plot(np.cumsum(pca.explained_variance_ratio_),label=sub+1)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative explained variance")
    plt.legend()
plt.show()
#%%
for sub in range(num_sub): 
    features_test[sub]=[]
    labels_test[sub]=[]
    if (sub+1)<10:
        file_test='S0%dE.mat'% (sub+1)
        mat_test=scipy.io.loadmat(file_test)
    else:
        file_test='S%dE.mat'% (sub+1)
        mat_test=scipy.io.loadmat(file_test)
#    print("Mat file read")
    data_test=mat_test['data']
    key_count=0
    for freq_count in range(freq.shape[0]): # loop for frequency
        tmp_feat=[]
#         lower=freq[freq_count]
#         if lower == freq[-1]:
#             break
#         higher=freq[freq_count+1]
#         if lower>higher:
#             continue
        lower=freq[freq_count,0]
        higher=freq[freq_count,1]
        for k in range(3):    
            cell_test=data_test[0][k]
            X_test=cell_test[0][0][0]
            X_test_filt=bandpass_filter(X_test,lowcut=lower, highcut=higher, fs=512, order=5)
            time_test=cell_test[0][0][1][0]
            if freq_count==0:
                labels_tmp_test=cell_test[0][0][2]
                labels_test[sub].extend(labels_tmp_test[0])    
            #Computing Spatial features for testing
            var=0
            for count in range(len(time_test)):
                tmp=get_feat(X_test_filt[var+int(4.5*sample_rate):var+8*sample_rate,:],np.array(spatial_filt[sub][key_count]))
                tmp_feat.append(tmp)
                var=time_test[count]
                
        tmp_feat=np.array(tmp_feat)
#        print("In testing ",key_count)
        key_count=key_count+1
        if freq_count==0:
            features_test[sub]=tmp_feat
        else:
            features_test[sub]=np.concatenate((features_test[sub],tmp_feat),axis=1)
    features_test[sub]=model[sub].transform(features_test[sub])
#    np.random.shuffle(features_test[sub])
print("Features for training and Testing computed")
#%%  Classifiers Assemble! 
''' Important to remember that each classifier is trained on the individual subjects separately.
This means that we are training 14(i.e. number of subjects) models
'''
#
nb_train_err=[]
nb_test_err=[]
rando_train_err=[]
rando_test_err=[]
knn_train_err=[]
knn_test_err=[]
reg_train_err=[]
reg_test_err=[]
xgb_train_err=[]
xgb_test_err=[]
nn_train_err=[]
nn_test_err=[]
for sub in range(num_sub):
    xtrain=features_train[sub]
    ytrain=labels_train[sub]
    xtest=features_test[sub]
    ytest=labels_test[sub]
    
    xgb_tr,xgb_tst=boost(xtrain,ytrain,xtest,depth=2)
    xgb_train_err.append(np.mean(ytrain!=xgb_tr))
    xgb_test_err.append(np.mean(ytest!=xgb_tst))
    
    rando_tr,rando_tst=random_forest(xtrain,ytrain,xtest,depth=2)
    rando_train_err.append(np.mean(ytrain!=rando_tr))
    rando_test_err.append(np.mean(ytest!=rando_tst))
    
    knn_tr,knn_tst=KNN(xtrain,ytrain,xtest,neighbor=4)
    knn_train_err.append(np.mean(ytrain!=knn_tr))
    knn_test_err.append(np.mean(ytest!=knn_tst))
    
    reg_tr,reg_tst=log_reg(xtrain,ytrain,xtest)
    reg_train_err.append(np.mean(ytrain!=reg_tr))
    reg_test_err.append(np.mean(ytest!=reg_tst))
    
    nn_tr,nn_tst=neural_net(xtrain,ytrain,xtest,neuron=10)
    nn_train_err.append(np.mean(ytrain!=nn_tr))
    nn_test_err.append(np.mean(ytest!=nn_tst))
    
    nb_tr,nb_tst=naive_bayes(xtrain,ytrain,xtest)
    nb_train_err.append(np.mean(ytrain!=nb_tr))
    nb_test_err.append(np.mean(ytest!=nb_tst))

some=[i+1 for i in range(14)]    
plt.scatter(some,rando_train_err,label="Train")
plt.scatter(some,rando_test_err,label="Test")
plt.title("Random Forest")
plt.xlabel("Subject")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.scatter(some,knn_train_err,label="Train")
plt.scatter(some,knn_test_err,label="Test")
plt.title("4 nearest neighbors")
plt.xlabel("Subject")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.scatter(some,reg_train_err,label="Train")
plt.scatter(some,reg_test_err,label="Test")
plt.title("Logistic regression")
plt.xlabel("Subject")
plt.ylabel("Error")
plt.legend()
plt.show()
#
#plt.plot(some,nn_train_err,label="Train")
#plt.plot(some,nn_test_err,label="Test")
#plt.title("Neural Network")
#plt.legend()
#plt.show()

plt.scatter(some,xgb_train_err,label="Train")
plt.scatter(some,xgb_test_err,label="Test")
plt.title("XGBoost")
plt.xlabel("Subject")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.scatter(some,nb_train_err,label='Train')
plt.scatter(some,nb_test_err,label='Test')
plt.legend()
plt.xlabel("Subject")
plt.ylabel("Error")
plt.title("Naive Bayes")
plt.show()


#    print("Train error:",np.mean(labels_train!=rando_tr),"Test error:",np.mean(labels_test!=rando_tst))


#
#knn_tr,knn_tst=KNN(features_train,labels_train,features_test,neighbor=4)
#print("Train error:",np.mean(labels_train!=knn_tr),"\tTest error:",np.mean(labels_test!=knn_tst))
#
#
#reg_tr,reg_tst=log_reg(features_train,labels_train,features_test)
#print("Train error:",np.mean(labels_train!=reg_tr),"\tTest error:",np.mean(labels_test!=reg_tst))
#
#
#rando_tr,rando_tst=random_forest(features_train,labels_train,features_test,depth=5)
#print("Train error:",np.mean(labels_train!=rando_tr),"Test error:",np.mean(labels_test!=rando_tst))
#
##net_tr,net_tst=neural_net(features_train,labels_train,features_test,neuron=20)
##print("Train error:",np.mean(labels_train!=net_tr),"\tTest error:",np.mean(labels_test!=net_tst))
#
#xgb_tr,xgb_tst=boost(features_train,labels_train,features_test,depth=10)
#print("Train error:",np.mean(labels_train!=xgb_tr),"\tTest error:",np.mean(labels_test!=xgb_tst))
#
#
#lda_tr,lda_tst=lda(features_train,labels_train,features_test)
#print("Train error:",np.mean(labels_train!=lda_tr),"\tTest error:",np.mean(labels_test!=lda_tst))


#%%
#from sklearn.svm import SVC
#X_train_svm = features_train
##y_svm = train_data[:,10]
##pca = PCA(n_components = X_train_svm.shape[1])
#pca = PCA(n_components=2)
#x_pca=pca.fit_transform(X_train_svm)
##var = pca.explained_variance_ratio_
##x_plot = []
##y_plot = []
##var_tot = 0
##for i in range(len(var)):
##    var_tot += var[i]
##    x_plot.append(i+1)
##    y_plot.append(var_tot)
##    
##plt.plot(x_plot,y_plot)
##plt.show()
#svm=SVC(kernel='rbf')
#svm.fit(x_pca,labels_train)
#
#plot_decision_regions(x_pca,np.array(labels_train),clf=svm,legend=2)
#plt.show()