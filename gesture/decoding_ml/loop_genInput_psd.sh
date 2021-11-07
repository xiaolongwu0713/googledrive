#!/bin/bash
# usage: ./loop_main_all.sh wind stride
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
	inputfile="H:/Long/data/gesture/preprocessing/Info.txt"
	echo "workstation"
fi


if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	inputfile="/Users/long/Documents/data/gesture/info/Info.txt"
	echo "longsMac"
fi


#declare -a sids
sids=()
while IFS= read -r line
do
 sid=${line%,*}
 sids+=("$sid")
 #echo $sid
 #echo ${sids[@]}
done < "$inputfile"
#echo ${sids[@]}

activ_channel='all'
for sid in ${sids[@]}
do
  echo "Start sid: $sid"
  network='eegnet'
  #for network in 'eegnet'# 'shallowFBCSPnet' 'deepnet' 'resnet'
  #do
     echo "*************Sid $sid on $network*************"
     #python main_all.py $sid $network fs wind stride
     python genInput_psd.py $sid $activ_channel
     echo "Training finish for sid: $sid on $network"
  #done
done

