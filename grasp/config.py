import numpy as np


MNE_LOGGING_LEVEL='ERROR' # or mne.set_log_level(verbose='ERROR'), then mne.set_log_level(return_old_level=True)

tmp_dir='/tmp/'
root_dir='/Users/long/BCI/python_scripts/googleDrive/' # this is project root
data_raw='/Volumes/Samsung_T5/seegData/' #raw data and processed data
#data_dir='/Users/long/BCI/data/grasp_data/PF6_SYF_2018_08_09_Simply/data/' # preprocessed data
data_dir='/content/drive/MyDrive/data/' # googleDrive
mode=1
processed_data=data_dir

fbands=[] #delta, theta, alpha,beta,gamma
fbands.append([0.5, 4])
fbands.append([4, 8])
fbands.append([8, 12])
fbands.append([13, 30])
fbands.append([60, 140])

ERD=[8,30]
ERS=[60,300]
# some cross module variables, you can import this variable as:
# import grasp.config as myVar, then myVar.preds=...
# OR, just make them global
#preds=[]
#targets=[]

# Lambda is used by skorch get_loss function
Lambda = 1e-6
preds=[]
targets=[]


def printVariables(variable_names):
    for k in variable_names:
        max_name_len = max([len(k) for k in variable_names])
        print(f'  {k:<{max_name_len}}:  {globals()[k]}')

if __name__ == "__main__":
    ks = [k for k in dir() if (k[:2] != "__" and k !='np' and not callable(globals()[k]))]
    #for k in ks:
    #    print(type(k))
    printVariables(ks)

