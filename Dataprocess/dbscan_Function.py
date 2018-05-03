from Dataprocess.cluster import Cluster
from sklearn.cluster import DBSCAN
from sklearn import mixture
from Dataprocess.trans_Data import *
from Dataprocess.get_Data import *
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle



def viewdbscan(Video_Id, frame_ind, path):
    DIM = 3
    ind = [1, 2, 3, 4, 5]

#读取数据
    test_Data = np.zeros((1, DIM+1))
    for i in range(35):
        for j in frame_ind:
            this_ind = np.where(getData(i, Video_Id)[:, 1] == j)[0]
            if(this_ind.shape[0] != 0):
                read_Data = getData(i, Video_Id)[this_ind[0]:this_ind[-1], ind]
                test_Data_new = np.hstack((read_Data[:, [0]], Qua2Eul(read_Data[:, [1, 2, 3, 4]])))
                test_Data = np.vstack((test_Data,test_Data_new))
        print(str(i))
    test_Data = np.delete(test_Data, 0, axis=0)
    print('LOAD_DATA_SUCCESS')


#聚类
    X = test_Data[:, [1, 2, 3]]
    t1 = np.floor(X.shape[0]*0.2).astype(int)
    t2 = 50
    Tr = [t1,t2]
    thresh_num = np.min(Tr)
    db = DBSCAN(eps=0.8, min_samples = t1).fit(X)
    skl_labels = db.labels_
    class_count = np.max(skl_labels) + 1
    noise = Cluster('Invalid', DIM)
    new_cluster = []
    for i in range(class_count):
        name = 'cluster-%d' % (i + 1)
        new_cluster.append(Cluster(name, DIM))


#画图
    fig = plt.figure()
    axis_proj = '%dd' % DIM
    ax = fig.add_subplot(111, projection=axis_proj)



    for i in range(0, len(skl_labels)):
        if not skl_labels[i] == -1:
            new_cluster[skl_labels[i]].add_point(X[i, :])
        else:
            noise.add_point(X[i, :])

    ax.scatter(noise.get_X(), noise.get_Y(), noise.get_Z(), marker='o', label=noise.name)
    print(noise.name)
    print(np.sum(noise.get_X()) / len(noise.get_X()))
    print(np.sum(noise.get_Y()) / len(noise.get_Y()))
    print(np.sum(noise.get_Z()) / len(noise.get_Z()))
    if class_count != 0:
        means = np.zeros([class_count,3])
    for i in range(class_count):
        ax.scatter(new_cluster[i].get_X(), new_cluster[i].get_Y(), new_cluster[i].get_Z(), marker='o',
                   label=new_cluster[i].name)
        print(new_cluster[i].name)
        print(np.sum(new_cluster[i].get_X()) / len(new_cluster[i].get_X()))
        print(np.sum(new_cluster[i].get_Y()) / len(new_cluster[i].get_Y()))
        print(np.sum(new_cluster[i].get_Z()) / len(new_cluster[i].get_Z()))
        means[i, 0] = np.sum(new_cluster[i].get_X()) / len(new_cluster[i].get_X())
        means[i, 1] = np.sum(new_cluster[i].get_Y()) / len(new_cluster[i].get_Y())
        means[i, 2] = np.sum(new_cluster[i].get_Z()) / len(new_cluster[i].get_Z())

    np.savetxt(path + 'Means_list.csv',means,delimiter=',')

    # ax.hold(False)
    ax.legend(loc='lower left')
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(r'Viewport DBSCAN Clustering', fontsize=18)
    # plt.show()
    plt.savefig(path + 'fig.png')


#GMM拟合
    for i in range(class_count):
        e = np.array(new_cluster[i].get_points())
        gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(e)
        with open(path + 'GMM_model'+ str(i)+'.pkl','wb') as f:
            pickle.dump(gmm,f)



