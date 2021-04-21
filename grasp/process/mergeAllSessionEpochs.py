import mne
import numpy as np
from grasp.config import activeChannels, stim,badtrials,data_raw,data_dir
import matplotlib.pyplot as plt

plot_dir = 'grasp/process/mergeAllSessionEpochs/'
sessions=4
movements=4

print("Load all movement epoch")
moveMix=[]
for movement in range(movements):
    moveMix.append([])
    for session in range(sessions):
        moveMix[movement].append([])
        filename='s'+str(session)+'move'+str(movement)+'BandsEpoch.fif'
        moveMix[movement][session]=mne.read_epochs(data_dir+filename)

print("Combine the same movement")
moves=[]
for i in range(movements):
    moves.append([])
    moves[i]=mne.concatenate_epochs(moveMix[i])

# reorder force,target and stim channel to the end
print("Reorder force, target and stimulateion channel to the end.")
for i in range(movements):
    ch_names=moves[i].ch_names
    forceIndex=ch_names.index('force') # 19
    targetIndex=ch_names.index('target') # 20
    stimIndex=ch_names.index('stimulation') # 21
    indexAll=list(range(len(ch_names)))
    indexAll.pop(forceIndex),indexAll.pop(forceIndex),indexAll.pop(forceIndex) # pop the same index. out: (19, 20, 21)
    indexAll.append(forceIndex),indexAll.append(targetIndex),indexAll.append(stimIndex) # append to the end
    moves[i].reorder_channels([ch_names[j] for j in indexAll])


print("Plot 4 movement force.")
for i in range(movements):
    fig,ax=plt.subplots()
    cubedata=moves[i].get_data() # (40, 117, 15001)
    ax.plot(cubedata[0,-1,:],label='stim'),ax.plot(cubedata[0,-2,:],label='target'),ax.plot(cubedata[0,-3,:],label='real')
    figname = root + plot_dir+'move'+str(i)+'force.png'
    fig.savefig(figname, dpi=400)
    plt.close(fig)

print("Saving 4 moves frequency band feature epoch.")
for i in range(movements):
    moves[i].save(data_dir+'move'+str(i)+'BandEpoch.fif', overwrite=True)
