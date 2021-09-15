import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
'''
T=0.001#
fs=1/T
L=fs*10#10s
t=np.arange(0,L)*T

x=1*np.sin(2*np.pi*60*t)
x+=2*np.sin(2*np.pi*100*t)
x+=4*np.sin(2*np.pi*200*t)
x+=5*np.sin(2*np.pi*300*t)

x1=butter_bandpass_filter(x,0.1,100,fs,5)
t=np.arange(x.shape[0])
plt.plot(t,x1)

'''




