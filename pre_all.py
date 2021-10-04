import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42

import sys
import socket

if socket.gethostname() == 'longsMac': # change this to your $HOSTNAME
    sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])
elif socket.gethostname() == 'workstation':
    sys.path.extend(['C:/cygwin/.....'])
else:
    print("Wrong hostname.")