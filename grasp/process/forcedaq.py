'''
Write a matlab function and call it from python.
convet force daq file to mat file
start matlab and run blow
>> filename='/Volumes/Samsung_T5/seegData/PF10/Force_Data/1-2.daq';
>> f = fopen(fullfile(filename));
>> data = fread(f,'double'); %(6039000,1)
>> save('/Volumes/Samsung_T5/seegData/PF10/Force_Data/1-2.mat','data');
'''
sid=16
import matlab.engine
eng = matlab.engine.start_matlab()
eng.convertDaqToMat(sid,nargout=0) # 10 is sid, nargout is necessary.


