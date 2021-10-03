#/bin/bash
cwd=`pwd`
echo $cwd
if [[ $cwd == C*  ]];then
	source /cygdrive/c/Users/wuxiaolong/venv/Scripts/activate
elif [[ $cwd == "/usr"  ]];then
	source /usr/local/venv/gesture/bin/activate
fi
for sid in 1 2 3 4 5 6 7 8 9
do
	python main_base.py $sid
done
