import numpy as np
import os

import Dataprocess.dbscan_Function as dbFunc

path = 'D:/testimg/'
video_id = 2


frame_cluster_ind = np.loadtxt(path + 'clusater.csv', delimiter = ',')
frame_cluster_num = frame_cluster_ind[np.argmax(frame_cluster_ind[:,1]),1]+1
frame_cluster_num = frame_cluster_num.astype(int)


# for i in range(frame_cluster_num):
for i in range(frame_cluster_num):
    this_cluster_frame_ind = np.array(np.where(frame_cluster_ind[:,1] == i))[0]
    if (this_cluster_frame_ind.shape[0] != 0):
        this_cluster_frame_path = path + str(i) + '_frame_cluster/'
        os.makedirs(this_cluster_frame_path)
        dbFunc.viewdbscan(video_id, this_cluster_frame_ind, this_cluster_frame_path)