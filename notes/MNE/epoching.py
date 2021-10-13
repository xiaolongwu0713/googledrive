# epoch in two ways.
# equal????
tmin=0
tmax=2
if 1==1:
    event1=events[(events[:,2]==0)]
    event2=events[(events[:,2]==1)]
    event3=events[(events[:,2]==2)]
    event4=events[(events[:,2]==3)]
    event5=events[(events[:,2]==4)]

    epoch1=mne.Epochs(raw, event1, tmin=tmin, tmax=tmax,baseline=None) # 1s rest + 2s task + 1s rest
    epoch2=mne.Epochs(raw, event2, tmin=tmin, tmax=tmax,baseline=None)
    epoch3=mne.Epochs(raw, event3, tmin=tmin, tmax=tmax,baseline=None)
    epoch4=mne.Epochs(raw, event4, tmin=tmin, tmax=tmax,baseline=None)
    epoch5=mne.Epochs(raw, event5, tmin=tmin, tmax=tmax,baseline=None)
    list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]
else:
    epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax,baseline=None)
    epoch1=epochs['0']
    epoch2=epochs['1']
    epoch3=epochs['2']
    epoch4=epochs['3']
    epoch5=epochs['4']
    list_of_epochs=[epoch1,epoch2,epoch3,epoch4,epoch5]