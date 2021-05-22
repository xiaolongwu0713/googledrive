'''
workflow:
1,run checkChannels.py to determine total channel number, bad channels, trigger channels
2,use tfAnalysis.py to choose the active channels.
3,use preprocess.py to check the trial force quality to determine the badtrials.
'''
import numpy as np
FS=[0]*999
chnNum=[0]*999 # total channels number
badChannels=[0]*999
anomalys=[0]*999
useChannels=[0]*999
triggerChannels=[0]*999 # triggerChannels is stim channel.
activeChannels=[0]*999
# Do not use : [[]]*999. Otherwise if you modify one sublist, you modify all sublist.
# https://stackoverflow.com/questions/8713620/appending-items-to-a-list-of-lists-in-python
badtrials=[[] for _ in range(99)]


# subject ID start with 1.
sids=[1,2,6,10,16] #6 is the first subject
#sids=[sid-1 for sid in sids] # [0, 1, 5, 9, 15]



#subject 1
FS[1]=2000
chnNum[1]=147
badChannels[1]=[14,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,145,146] #19
# not entirelly noise, but very unlikely
anomalys[1]=[98]
useChannels[1]=[item for item in [*range(chnNum[1])] if item not in badChannels[1] if item not in anomalys[1]] # 127
#matlab: useChannels=[1:14,16:30,47:145]
#aa=[*range(0,14)]+[*range(15,30)]+[*range(46,145)]. Same as mine.
triggerChannels[1]=38 # 37-42 all trigger channels
activeChannels[1]=[73,74,92,95,111,116]# 6. Very bad result.
badtrials[1].append([])
badtrials[1].append([])
badtrials[1].append([])
badtrials[1].append([])


# subject 2
#SubInfo.UseChn=[1:15,17:30,40:146]; % H9 is missing, create one virtual channel H9 for P25
#SubInfo.EmgChn=[147:148];
#SubInfo.TrigChn=[32:36];
matlabUseChn=[*range(0,15)]+[*range(16,30)]+[*range(39,145)] # 135
matlabTriggerChn=[*range(31,36)]
FS[2]=2000
chnNum[2]=148
# [15, 30, 31, 32, 33, 34, 35, 36, 37, 38, 145, 146, 147]
#mine: [15,31,32,33,34, 35,36,37,38,146,147]
badChannels[2]=[15, 30, 31, 32, 33, 34, 35, 36, 37, 38, 145, 146, 147] #13
# not entirelly noise, but very unlikely
anomalys[2]=[]
useChannels[2]=[item for item in [*range(chnNum[2])] if item not in badChannels[2] if item not in anomalys[2]] # 128
#matlab: useChannels=[1:14,16:30,47:145]
#aa=[*range(0,14)]+[*range(15,30)]+[*range(46,145)]. Same as mine.
triggerChannels[2]=31 # 31-38 all trigger channels
activeChannels[2]=[*range(61,69)] +[76,77,78]+ [*range(126,131)]#16 channels: 127-118,69,58-55
# specify the badtrials for all 4 movements.
badtrials[2].append([])
badtrials[2].append([])
badtrials[2].append([])
badtrials[2].append([])



# subject 6
FS[6]=2000
useChannels[6]=np.concatenate((np.arange(0,15),np.arange(16,29),np.arange(37,119))) #110 channels
triggerChannels[6]=29
# activeChannels[6]: 19 channels
activeChannels[6] = [i-1 for i in [8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107,108, 109, 110]]#19 chns. 111 is force channel, index start from 1

# badtrials[sid][trialNum]
badtrials[6].append([])
badtrials[6][0]=[i-1 for i in [12, 13, 21, 22, 23, 24, 26]]
badtrials[6].append([])
badtrials[6][1]=[i-1 for i in [23, 24, 28]]
badtrials[6].append([])
badtrials[6][2]=[i-1 for i in [3, 4, 8, 11, 16, 21, 22, 23, 24, 25, 29]]
badtrials[6].append([])
badtrials[6][3]=[i-1 for i in [4, 21, 24, 25, 29]]
stim=30 - 1

# subject 10
#matlab:
#SubInfo.UseChn=[1:15,17:31,43:114];
#SubInfo.EmgChn=[117:118];
#SubInfo.TrigChn=[35:39];
matlabUseChn=[*range(0,15)]+[*range(16,31)]+[*range(42,114)] # same as python
matlabTriggerChn=[*range(34,39)]
FS[10]=2000
chnNum[10]=122

badChannels[10]=[15,31,32,33,34,35,36,37,38,39,40,41,114,115,116,117,118,119,120,121] #20
# not entirelly noise, but very unlikely
anomalys[10]=[]
useChannels[10]=[item for item in [*range(chnNum[10])] if item not in badChannels[10] if item not in anomalys[10]] # 128
triggerChannels[10]=34 # 31-38 all trigger channels
# determine the active channel with tf plot.
activeChannels[10]=[37]+[*range(62,71)]+[81] # 11 chns
# specify the badtrials for all 4 movements.
badtrials[10].append([26])
badtrials[10].append([])
badtrials[10].append([1])
badtrials[10].append([2])

#subject 16
#SubInfo.UseChn=[1:19,21:37,54:207];
#SubInfo.EmgChn=[210:211];
#SubInfo.TrigChn=[46:48];
matlabUseChn=[*range(0,19)]+[*range(20,37)]+[*range(53,207)] # same as python
matlabTriggerChn=[*range(45,48)]
FS[16]=2000
chnNum[16]=217

badChannels[16]=[19,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,209,216] # 38,39,40,50,51,53 were added from matlabUseChn
# not entirelly noise, but very unlikely
anomalys[16]=[]
useChannels[16]=[item for item in [*range(chnNum[16])] if item not in badChannels[16] if item not in anomalys[16]] # 128
triggerChannels[16]=45 # 31-38 all trigger channels
# determine the active channel with tf plot.
activeChannels[16]=[*range(15,21)]+[23]+[*range(29,36)]+[37]+[*range(39,45)]+\
                   [52,53]+[*range(61,70)]+[77,78,79]+[96,97]+[104,105]+[*range(161,189)]#15 channels: 127-118,69,58-55
# specify the badtrials for all 4 movements.
badtrials[16].append([])
badtrials[16].append([])
badtrials[16].append([])
badtrials[16].append([])
