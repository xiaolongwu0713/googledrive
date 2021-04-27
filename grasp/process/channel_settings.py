import numpy as np
FS=[0]*999
chnNum=[0]*999 # total channels number
badChannels=[0]*999
anomalys=[0]*999
useChannels=[0]*999
triggerChannels=[0]*999 # triggerChannels is stim channel.
activeChannels=[0]*999
badtrials=[[]]*999


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
activeChannels[2]=[*range(55,59)] +[69]+ [*range(118,128)]#15 channels: 127-118,69,58-55
# specify the badtrials for all 4 movements.
badtrials[2].append([])
badtrials[2].append([])
badtrials[2].append([])
badtrials[2].append([])


# subject 6
FS[6]=2000
useChannels[6]=np.concatenate((np.arange(0,15),np.arange(16,29),np.arange(37,119))) #110 channels
triggerChannels[6]=29
activeChannels[6] = [i-1 for i in [8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 62, 63, 69, 70, 105, 107,108, 109, 110]]# 111 is force channel, index start from 1

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
