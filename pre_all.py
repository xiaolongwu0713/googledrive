import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42

import sys,os, re
import socket

location=os.getcwd()
if re.compile('/content/drive').match(location): # google colab
    pass
