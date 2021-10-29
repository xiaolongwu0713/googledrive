#/bin/bash
cwd=`pwd`
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
	echo "workstation"
fi


if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	echo "longsMac"
fi

for sid in 2 3 4 5 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26 29 30 31 32 34 35 41
do
  for network in 'eegnet' 'shallowFBCSPnet' 'deepnet' 'resnet'
  do
      python main_all.py $sid $network
  done
done
