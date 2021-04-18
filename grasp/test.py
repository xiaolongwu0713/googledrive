import matplotlib.pyplot as plt
import numpy as np
from grasp.utils import gen3DOnTheFly, rawDataSimply

dataiter=gen3DOnTheFly()
x,y=next(dataiter)
print(x.shape)
print(y.shape)
x,y=next(dataiter)
print(x.shape)
print(y.shape)