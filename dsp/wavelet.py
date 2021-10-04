import matplotlib.pyplot as plt
import pywt

w = pywt.Wavelet('bior6.8')
(dec_lo, dec_hi, rec_lo, rec_hi)=w.filter_bank

plt.plot(dec_lo)
plt.plot(dec_hi)
plt.plot(rec_lo)
plt.plot(rec_hi)

# copied from http://wavelets.pybytes.com/wavelet/bior6.8/   Coefficients/Decomposition low-pass filter
dec_lo2=[0.0,0.0019088317364812906,-0.0019142861290887667,-0.016990639867602342,0.01193456527972926,0.04973290349094079,-0.07726317316720414,
-0.09405920349573646,0.4207962846098268,0.8259229974584023,0.4207962846098268,-0.09405920349573646,-0.07726317316720414,
0.04973290349094079,0.01193456527972926,-0.016990639867602342,-0.0019142861290887667,0.0019088317364812906]
# they are equal
assert(dec_lo==dec_lo2)

(phi_d, psi_d, phi_r, psi_r, x) = w.wavefun(level=5)












