import numpy as np
import os

filepath = './Database/'
videodic = ['Venise-s-AJRFQuAtE', 'Paris-sJxiPiAaB4k', 'Elephant-training-2bpICIClAIg', 'Diving-2OzlksZBTiA', 'Rollercoaster-8lsB-P8nGSM', 'Timelapse-CIw8R8thnm8']

#方便读取数据
def FindDirList(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                h = os.path.split(m)
                list.append(h[1])
    return list

def readfile(path):
    datapath = path + os.listdir(path)[0]
    data = np.loadtxt(datapath, dtype=float, delimiter=' ')
    return data

def getData(usrs_id,video_id):
    usrlist = FindDirList(filepath)
    data = np.zeros((1,6),dtype='float')
    videolist = FindDirList(filepath + usrlist[usrs_id] + '/test0')
    if videodic[video_id] in videolist:
        localpath = filepath + usrlist[usrs_id] + '/test0/' + videodic[video_id] + '/'
        data = np.vstack((data,readfile(localpath)))
    data = np.delete(data,0,axis=0)
    return data

