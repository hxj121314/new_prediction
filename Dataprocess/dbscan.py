from Dataprocess.cluster import Cluster
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from Dataprocess.trans_Data import *
from Dataprocess.get_Data import *

DIM = 3

#生成随机数据
# centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 0]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
# X = StandardScaler().fit_transform(X)
ind = [2,3,4,5]
fram_ind_start = 11
fram_ind_end = 20
for i in range(50):
    if i == 0:
        test_Data = Qua2Eul(getData(i, 0)[fram_ind_start:fram_ind_end, ind])
    else:
        test_Data_new = Qua2Eul(getData(i, 0)[fram_ind_start:fram_ind_end, ind])
        test_Data = np.vstack((test_Data,test_Data_new))
X = test_Data

#dbscan
db = DBSCAN(eps=0.8, min_samples=80, metric='manhattan').fit(X)
skl_labels = db.labels_
class_count = np.max(skl_labels) + 1
noise = Cluster('Invalid', DIM)
new_cluster = []
for i in range(class_count):
    name = 'cluster-%d' % (i+1)
    new_cluster.append(Cluster(name, DIM))


#画图
fig = plt.figure()
axis_proj = '%dd' % DIM
ax = fig.add_subplot(111, projection = axis_proj)
for i in range(0, len(skl_labels)):
    if not skl_labels[i] == -1:
        new_cluster[skl_labels[i]].add_point(X[i, :])
    else:
        noise.add_point(X[i, :])

ax.hold(True)
ax.scatter(noise.get_X(), noise.get_Y(), noise.get_Z(), marker = 'o', label = noise.name)
for i in range(class_count):
    ax.scatter(new_cluster[i].get_X(), new_cluster[i].get_Y(), new_cluster[i].get_Z(), marker='o', label=new_cluster[i].name)

ax.hold(False)
ax.legend(loc='lower left')
ax.grid(True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(r'Viewport DBSCAN Clustering', fontsize=18)
plt.show()

