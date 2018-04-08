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

DIM = 3

#生成随机数据
# centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 0]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
# X = StandardScaler().fit_transform(X)
ind = [1,2,3,4,5]
fram_ind_start = 11
fram_ind_end = 20
for i in range(50):
    if i == 0:
        this_ind_start = np.where(getData(i, 0)[:,1]==fram_ind_start)[0][0]
        this_ind_end = np.where(getData(i, 0)[:, 1] == fram_ind_end)[0][0]
        read_Data = getData(i, 0)[this_ind_start:this_ind_end, ind]
        test_Data = np.hstack((read_Data[:,[0]], Qua2Eul(read_Data[:,[1,2,3,4]])))
    else:
        this_ind_start = np.where(getData(i, 0)[:, 1] == fram_ind_start)[0][0]
        this_ind_end = np.where(getData(i, 0)[:, 1] == fram_ind_end)[0][0]
        read_Data_new = getData(i, 0)[this_ind_start:this_ind_end, ind]
        test_Data_new = np.hstack((read_Data_new[:, [0]], Qua2Eul(read_Data_new[:, [1, 2, 3, 4]])))
        test_Data = np.vstack((test_Data,test_Data_new))
X = test_Data[:,[1,2,3]]
np.savetxt('E:\\viewframe.csv',test_Data,delimiter=',')
print(test_Data)

print(X.shape)

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

########################################
# #SGM regression
# e=np.array(new_cluster[1].get_points()).T
# start=[0,0,0]
# for i in range(len(new_cluster[1].get_points())):
#       start=start+e.T[i,:]
#
# start=np.mat(start)
# mu=start/len(new_cluster[1].get_points())
# mu=np.mat(mu).T
# sigma=np.cov(e)
# sigma=np.mat(sigma)
# print(mu)
# print(sigma)

######################################
# e1=np.array(new_cluster[1].get_points()).T
# start1=[0,0,0]
# for i in range(len(new_cluster[0].get_points())):
#       start1=start1+e1.T[i,:]
#
# start1=np.mat(start1)
# mu1=start1/len(new_cluster[0].get_points())
# mu1=np.mat(mu1).T
# sigma1=np.cov(e1)
# sigma1=np.mat(sigma1)
# print(mu)
# print(sigma)

#######################################
#GMM regression
# e=np.array(new_cluster[0].get_points())
# gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(e)
# print(gmm.weights_)
# e1=np.array(new_cluster[0].get_points())
# print(gmm.predict(e1[4:10,:]))
# print(gmm.predict_proba(e1[4:10,:]))
# print(-gmm.score(e1[4:10,:]))
# print(-gmm.score_samples(e1[4:10,:]))


# ax.hold(True)
ax.scatter(noise.get_X(), noise.get_Y(), noise.get_Z(), marker = 'o', label = noise.name)
print(noise.name)
print(np.sum(noise.get_X())/len(noise.get_X()))
print(np.sum(noise.get_Y())/len(noise.get_Y()))
print(np.sum(noise.get_Z())/len(noise.get_Z()))
for i in range(class_count):
    ax.scatter(new_cluster[i].get_X(), new_cluster[i].get_Y(), new_cluster[i].get_Z(), marker='o', label=new_cluster[i].name)
    print(new_cluster[i].name)
    print(np.sum(new_cluster[i].get_X()) / len(new_cluster[i].get_X()))
    print(np.sum(new_cluster[i].get_Y()) / len(new_cluster[i].get_Y()))
    print(np.sum(new_cluster[i].get_Z()) / len(new_cluster[i].get_Z()))

# ax.hold(False)
ax.legend(loc='lower left')
ax.grid(True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title(r'Viewport DBSCAN Clustering', fontsize=18)
plt.show()

