import hdf5storage
import matplotlib.pyplot as plt
import scipy.io
filename='/Users/long/Documents/BCI/matlab_scripts/ruijin/gonogo/pac_result1.mat'
#mat2=hdf5storage.loadmat(filename)

mat=scipy.io.loadmat(filename)
raw=mat['result']
del mat

chn_number=10
freq4phase = [2,4,6,8,10]
freq4power = range(20,110,10) #20:10:100;


channel1=list(range(10)) #[i-1 for i in [1,2]]
channel2=list(range(10)) #[i for i in range(63)]
data={}
for i in channel1:
    #data.append([])
    for j in channel2:
        for f in freq4phase:
            com=str('c'+str(i+1)+'_'+str(j+1)+'_'+str(f))
            data[com]=raw[com][0][0]
del raw


f_phase=mat['B']['freqs_phase'][0][0][0] # 4
f_amp=mat['B']['freqs_amp'][0][0][0] # 20

# 63 channels * 63 channels * 4 pac/channel-pair = 15876 plots
for c1 in channel1:
    for c2 in channel2:
        one_plot=data[c1][c2][0] # (4, 20, 2500)

# plot 20 pac/page
vmin=-4
vmax=4
fig,ax=plt.subplots(10,5,figsize=(6,3))
for i in channel1:
    # i-th image
    for j in channel2:
        #j-th row
        for k,f in enumerate(freq4phase):
            #f-th column
            com=str('c'+str(i+1)+'_'+str(j+1)+'_'+str(f))
            datai=data[com]
            axi=ax[j,k]
            im0 = axi.imshow(datai, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axi.plot(datai)


haha=[1,2,3,4,5]
for m,n in enumerate(haha):
    print(m,n)












