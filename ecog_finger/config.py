import os, re


sids=[1,2,3,4,5,6,7,8,9]

location=os.getcwd()
if re.compile('/Users/long/').match(location):
    data_dir='/Users/long/Documents/data/ecog/' # preprocessed data
    tmp_data_dir='/Users/long/Documents/data/gesture/'
    root_dir = '/Users/long/BCI/python_scripts/googleDrive/'  # this is project root
elif re.compile('/content/drive').match(location):
    data_dir='/content/drive/MyDrive/data/ecog/' # googleDrive
    root_dir='/content/drive/MyDrive/' # googleDrive
elif re.compile('C:').match(location):
    data_dir='C:/Users/wuxiaolong/Desktop/BCI/data/' # googleDrive
    root_dir='C:/Users/wuxiaolong/Desktop/BCI/googledrive/' # googleDrive
