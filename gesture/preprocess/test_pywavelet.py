import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import scale2frequency
from scipy.signal import chirp

# chirp signal
t = np.linspace(0, 10, 10000) # fs=1000, 10s
sampling_period=0.001
chirpdata = chirp(t, f0=1, f1=100, t1=10, method='linear')
#plt.plot(t, chirpdata)

# choose your wavelet and plot to examine it
w1 = pywt.ContinuousWavelet('cmor1.5-1.0')
w2 = pywt.ContinuousWavelet('cmor1.5-50.0')
wavelet_function, x_values=w1.wavefun()
plt.plot(x_values,wavelet_function)


# cmorB-C
# scale to frequency: f=center_frequency/(scale*sampling_period), Reverse the calculation to get the desired frequency:
# Reverse the calculation: scale=center_frequency/(desired_frequency*sampling_period)
scales=[1.0/(f*sampling_period) for f in np.arange(1,150,2)] # np.arange(1,150,2) is your desired frequencies
# verification
mwavelet='cmor1.5-1.0'
frequencies=[scale2frequency(mwavelet, scale)/sampling_period for scale in scales]
[coefficients, frequencies] = pywt.cwt(chirpdata, scales, mwavelet, sampling_period)
power = (abs(coefficients)) ** 2
vmin=-4
vmax=4
fig, ax = plt.subplots()
im0=ax.imshow(power,origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
ax.set_aspect('auto')
# frequency precision decrease as frequency increase


# increase bandwidth fb to increase frequency precision
mwavelet='cmor2.5-1.0'
scales=[1.0/(f*sampling_period) for f in np.arange(1,150,2)] # np.arange(1,150,2) is your desired frequencies
# verification
frequencies=[scale2frequency(mwavelet, scale)/sampling_period for scale in scales]
[coefficients, frequencies] = pywt.cwt(chirpdata, scales, mwavelet, sampling_period)
power = (abs(coefficients)) ** 2
vmin=-4
vmax=4
fig, ax = plt.subplots()
im0=ax.imshow(power,origin='lower',cmap='RdBu_r',vmin=vmin, vmax=vmax)
ax.set_aspect('auto')



# coutour line
fig, ax = plt.subplots()
ax.contour(range(power.shape[1]), range(power.shape[0]), power, extend='both',cmap=plt.cm.seismic)










