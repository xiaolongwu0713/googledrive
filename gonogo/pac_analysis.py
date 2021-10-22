import hdf5storage
import matplotlib.pyplot as plt
import scipy.io
import os
import numpy as np
base_dir='/Users/long/Documents/BCI/matlab_scripts/ruijin/gonogo/'
filename=base_dir+'pac_result2.mat'
save_dir=base_dir+'pac_plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#mat2=hdf5storage.loadmat(filename)

mat=scipy.io.loadmat(filename)
raw=mat['result'] # raw is a np structured arrays: https://numpy.org/doc/stable/user/basics.rec.html
del mat

channel1 = list(range(10))  # [i-1 for i in [1,2]]
channel2 = list(range(10))  # [i for i in range(63)]
freq4phase = [2,4,6,8,10]
freq4power = range(20,110,10) #20:10:100;

# show column:
#raw.dtype
len(raw.dtype.fields)==len(channel1)*len(channel2)*5

# plot 20 pac/page
cue_on_idex=5
cue_off_idex=35
vmin=-4
vmax=4
plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
plt.rc('ytick', labelsize=4)    # fontsize of the tick labels
#plt.rc('y', pad=0)
fig, ax = plt.subplots(10, 5, figsize=(6, 3), sharex=True)
for c1 in channel1:
    # i-th image
    if c1 != 0:
        ax=fig.subplots(10, 5, sharex=True)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.0)
    for c2 in channel2:
        #j-th row
        for k,f in enumerate(freq4phase):
            #f-th column
            com=str('c'+str(c1+1)+'_'+str(c2+1)+'_'+str(f))
            datai=raw[com][0][0]
            axi=ax[c2,k]
            im0 = axi.imshow(datai, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axi.axvline(x=cue_on_idex, linewidth=1, color='r', linestyle='--')
            axi.axvline(x=cue_off_idex, linewidth=1, color='r', linestyle='--')
            axi.yaxis.set_tick_params(labelsize=4)  # ticklable smaller小坐标轴
            axi.tick_params(axis='y', which='major', pad=0.0)  # space between ticklabel and yaxis，坐标轴紧凑
             # orleft=0.1,bottom=0.1,right=0.9,top=0.9,子图靠近。
    fig.suptitle("Channel " + str(c1)+ ". Low frequency(column):2,4,6,8,10 Hz; High f(row): 20:20:100;",fontsize=8)
    save_file=save_dir+com+'.pdf'
    fig.savefig(save_file)
    fig.clear()












