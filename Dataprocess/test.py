from Dataprocess.cluster import Cluster
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from Dataprocess.trans_Data import *
from Dataprocess.get_Data import *

# DIM = 3

#生成随机数据
# centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 0]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
# X = StandardScaler().fit_transform(X)
# Video_Id = 4
# ind = [1,2,3,4,5]
# iii = 2
# fram_ind_start = 11+50*iii
# fram_ind_end = 20+50*iii
#
# i =0
# this_ind = np.where(getData(i, Video_Id)[:, 1] == fram_ind_start)[0]
# read_Data = getData(i, Video_Id)[this_ind[0]:this_ind[-1], ind]
# test_Data = np.hstack((read_Data[:,[0]], Qua2Eul(read_Data[:,[1,2,3,4]])))
# print(test_Data.shape)

# s = np.array([1,3,4,5])
# this_ind = np.where(s == 6)[0]
# print(this_ind.shape[0])
# print(np.max([10,15]))


A = np.array([12,4,3,0,2])
B = A >1
B = np.sum(B)/len(B)
print(B)