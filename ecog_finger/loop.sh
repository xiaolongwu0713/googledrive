#/bin/bash
source /usr/local/venv/gesture/bin/activate
for sid in 1 2 3 4 5 6 7 8 9
do
	python main_base.py $sid
done
