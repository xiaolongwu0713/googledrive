import matplotlib.pyplot as plt
import pywt

w = pywt.Wavelet('bior6.8')
(dec_lo, dec_hi, rec_lo, rec_hi)=w.filter_bank

plt.plot(dec_lo)
plt.plot(dec_hi)
plt.plot(rec_lo)
plt.plot(rec_hi)

