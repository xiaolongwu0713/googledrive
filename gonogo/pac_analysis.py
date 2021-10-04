import hdf5storage
import scipy.io
filename='/Users/long/Documents/BCI/matlab_scripts/ruijin/gonogo/a2.mat'
mat2=hdf5storage.loadmat(filename)
mat=scipy.io.loadmat(filename)

chn_number=63
channel1=[i-1 for i in [1,2]]
channel2=[i for i in range(63)]
data=[]
for i in channel1:
    data.append([])
    for j in channel2:
        com=str('c'+str(i+1)+'_'+str(j+1))
        data[i-1].append(mat['A'][com][0])

f_phase=mat['B']['freqs_phase'][0][0][0]
f_amp=mat['B']['freqs_amp'][0][0][0]

for c1 in channel1:
    for c2 in channel2:
        one_plot=data[c1][c2][0]

