'''
check the brain area each electrode located.
'''

import sys
import h5py
import hdf5storage

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/long/Documents/BCI/python_scripts/googleDrive'])

from mne.time_frequency import tfr_morlet
from sklearn import preprocessing

from grasp.process.utils import get_trigger, genSubTargetForce, getRawData, getMovement, getForceData, \
    get_trigger_normal, getRawDataInEdf, getMovement_sid10And16
import numpy as np
import mne
import matplotlib.pyplot as plt
from grasp.config import *
from grasp.process.channel_settings import *


sid=16
sessions=4
movements=4

#plot_dir=root_dir+'grasp/process/result/'
plot_dir=data_dir + 'PF' + str(sid) +'/locations/'
import os
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

#SignalChanel_Electrode_Registration.mat: Electrodes number, e.g.: 1,2,3,....110
#Electrode_Name_SEEG.mat: Electrode name, g.g.:  'POL B9'
#WholeCortex.mat: a lot of number, have no idea
#electrodes_Final.mat: List of code: A1, A2, A3...B1,B2...C1,C2...

import matlab.engine
mat = matlab.engine.start_matlab()
if sid==16:
    matfile='/Volumes/Samsung_T5/seegData/PF'+str(sid)+'/BrainElectrodes/electrodes_Final_Norm.mat'
else:
    matfile='/Volumes/Samsung_T5/seegData/PF'+str(sid)+'/BrainElectrodes/electrodes_Final_Anatomy_wm.mat'

#locations=mat.read_h5file_electrodes_natomy(matfile,nargout=1) # call function in Matlab to return a list of all brain areas.
#Or you can read the file directly with mat.load as a dict.
f = mat.load(matfile, nargout=1) #f.keys(): dict_keys(['AsegWM', 'elec_Info_Final_wm'])
#then get the variable by descending the dict
locations=f['elec_Info_Final_wm']['ana_label_name'] # return a list.
set(locations)

'''
subject 1
{'Right-Amygdala',
 'Right-Hippocampus',
 'ctx--rh-parsorbitalis',
 'ctx-rh-caudalmiddlefrontal',
 'ctx-rh-fusiform',
 'ctx-rh-inferiortemporal',
 'ctx-rh-insula',
 'ctx-rh-lateralorbitofrontal',
 'ctx-rh-medialorbitofrontal',
 'ctx-rh-middletemporal',
 'ctx-rh-parahippocampal',
 'ctx-rh-parsopercularis',
 'ctx-rh-parsorbitalis',
 'ctx-rh-parstriangularis',
 'ctx-rh-precentral',
 'ctx-rh-rostralanteriorcingulate',
 'ctx-rh-rostralmiddlefrontal',
 'ctx-rh-supramarginal',
 'wm-rh-caudalmiddlefrontal',
 'wm-rh-fusiform',
 'wm-rh-inferiortemporal',
 'wm-rh-insula',
 'wm-rh-lateralorbitofrontal',
 'wm-rh-medialorbitofrontal',
 'wm-rh-middletemporal',
 'wm-rh-parahippocampal',
 'wm-rh-parsopercularis',
 'wm-rh-rostralanteriorcingulate',
 'wm-rh-rostralmiddlefrontal',
 'wm-rh-superiorfrontal',
 'wm-rh-superiortemporal',
 'wm-rh-supramarginal'}
'''

'''
subject 2
{'Left-Amygdala',
 'Left-Hippocampus',
 'Left-Inf-Lat-Vent',
 'Left-UnsegmentedWhiteMatter',
 'Unknown',
 'ctx--lh-supramarginal',
 'ctx-lh-fusiform',
 'ctx-lh-inferiorparietal',
 'ctx-lh-inferiortemporal',
 'ctx-lh-isthmuscingulate',
 'ctx-lh-middletemporal',
 'ctx-lh-parahippocampal',
 'ctx-lh-posteriorcingulate',
 'ctx-lh-precuneus',
 'ctx-lh-superiorparietal',
 'ctx-lh-superiortemporal',
 'ctx-lh-supramarginal',
 'wm-lh-fusiform',
 'wm-lh-inferiorparietal',
 'wm-lh-inferiortemporal',
 'wm-lh-insula',
 'wm-lh-isthmuscingulate',
 'wm-lh-middletemporal',
 'wm-lh-parahippocampal',
 'wm-lh-posteriorcingulate',
 'wm-lh-precuneus',
 'wm-lh-superiorparietal',
 'wm-lh-superiortemporal',
 'wm-lh-supramarginal'}
 '''
'''
Subject 6
{'Left-UnsegmentedWhiteMatter',
 'Unknown',
 'ctx--lh-postcentral',
 'ctx--lh-precentral',
 'ctx-lh-caudalmiddlefrontal',
 'ctx-lh-paracentral',
 'ctx-lh-postcentral',
 'ctx-lh-posteriorcingulate',
 'ctx-lh-precentral',
 'ctx-lh-superiorfrontal',
 'wm-lh-caudalmiddlefrontal',
 'wm-lh-paracentral',
 'wm-lh-posteriorcingulate',
 'wm-lh-precentral',
 'wm-lh-superiorfrontal'}
 '''

'''
subject 10
{'Unknown',
 'ctx--rh-postcentral',
 'ctx--rh-precentral',
 'ctx--rh-superiorparietal',
 'ctx-rh-inferiorparietal',
 'ctx-rh-paracentral',
 'ctx-rh-postcentral',
 'ctx-rh-precentral',
 'ctx-rh-precuneus',
 'ctx-rh-superiorparietal',
 'ctx-rh-supramarginal',
 'wm-rh-inferiorparietal',
 'wm-rh-postcentral',
 'wm-rh-precentral',
 'wm-rh-precuneus',
 'wm-rh-superiorparietal',
 'wm-rh-supramarginal'}
 '''

'''
Subject 16
{'Right-Cerebral-White-Matter',
 'Unknown',
 'WM-hypointensities',
 'ctx_rh_G_Ins_lg_and_S_cent_ins',
 'ctx_rh_G_and_S_cingul-Mid-Ant',
 'ctx_rh_G_and_S_cingul-Mid-Post',
 'ctx_rh_G_and_S_paracentral',
 'ctx_rh_G_and_S_subcentral',
 'ctx_rh_G_cingul-Post-dorsal',
 'ctx_rh_G_front_inf-Opercular',
 'ctx_rh_G_front_inf-Triangul',
 'ctx_rh_G_front_middle',
 'ctx_rh_G_front_sup',
 'ctx_rh_G_insular_short',
 'ctx_rh_G_pariet_inf-Angular',
 'ctx_rh_G_pariet_inf-Supramar',
 'ctx_rh_G_parietal_sup',
 'ctx_rh_G_postcentral',
 'ctx_rh_G_precentral',
 'ctx_rh_G_precuneus',
 'ctx_rh_G_temp_sup-G_T_transv',
 'ctx_rh_G_temp_sup-Lateral',
 'ctx_rh_G_temp_sup-Plan_tempo',
 'ctx_rh_Lat_Fis-ant-Vertical',
 'ctx_rh_Lat_Fis-post',
 'ctx_rh_S_central',
 'ctx_rh_S_cingul-Marginalis',
 'ctx_rh_S_circular_insula_inf',
 'ctx_rh_S_circular_insula_sup',
 'ctx_rh_S_front_middle',
 'ctx_rh_S_front_sup',
 'ctx_rh_S_intrapariet_and_P_trans',
 'ctx_rh_S_parieto_occipital',
 'ctx_rh_S_postcentral',
 'ctx_rh_S_precentral-inf-part',
 'ctx_rh_S_precentral-sup-part',
 'ctx_rh_S_subparietal',
 'ctx_rh_S_temporal_sup'}
 '''

