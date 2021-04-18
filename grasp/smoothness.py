import numpy as np
import matplotlib.pyplot as plt


x1=np.array([0.05] * 10)
x2=np.array([5*t for t in np.arange(1,11)])-4.95
x3=np.array([45.05]* 10)
x4=np.array([0.05] * 10)

x=np.concatenate((x1,x2,x3,x4),axis=0)

xnoise1 = np.random.normal(0,5,40) + x
xnoise2 = np.random.normal(0,10,40) + x
xnoise3 = np.random.normal(0,20,40) + x
plt.plot(x)
plt.plot(xnoise1,label=str())
plt.plot(xnoise2)
plt.plot(xnoise3)
np.std(np.diff(xnoise1))
np.std(np.diff(xnoise2))
np.std(np.diff(xnoise3))
