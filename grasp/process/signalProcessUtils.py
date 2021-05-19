from scipy.signal import butter, lfilter, freqz

def butter_lowpass_filter(data, cutoff, fs, order=5):
    #b, a = butter_lowpass(cutoff, fs, order=order)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
# filter analysis:  https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
#b, a = butter(order, normal_cutoff, btype='low', analog=False)
# Plot the frequency response.
#w, h = freqz(b, a, worN=8000)
#plt.subplot(2, 1, 1)
#plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
#plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#plt.axvline(cutoff, color='k')
#plt.xlim(0, 0.5*fs)
#plt.title("Lowpass Filter Frequency Response")
#plt.xlabel('Frequency [Hz]')
#plt.grid()

def getIndex(fMin, fMax, fstep, freq):
    freqs = [*range(fMin, fMax, fstep)]
    distance = [abs(fi - freq) for fi in freqs]
    index = distance.index(min(distance))
    return index
