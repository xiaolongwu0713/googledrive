#!/bin/bash
cwd=`pwd`
if [[ $HOSTNAME == "workstation"  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
	echo "workstation"
fi


if [[ $HOSTNAME == "longsMac"  ]];then
	source /usr/local/venv/gesture/bin/activate
	echo "longsMac"
fi

inputfile="H:/Long/data/gesture/preprocessing/Info.txt"
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

for sid in ${sids[@]}
#for sid in 10
do
  echo "Start sid: $sid"
  for network in 'eegnet'# 'shallowFBCSPnet' 'deepnet' 'resnet'
  do
     echo "*************Sid $sid on $network*************"
     #python main_all.py $sid $network fs wind stride
     python main_all.py $sid $network 1000 $1 $2
     echo "Training finish for sid: $sid on $network" 
  done
done

