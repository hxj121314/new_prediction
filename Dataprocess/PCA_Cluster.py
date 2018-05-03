# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import preprocessing
from matplotlib import pyplot as plt



stf = 140
enf = 160

path = 'D:/testimg'
result = os.listdir(path)
result.sort(key= lambda x:int(x[:-4]))
imlist = [os.path.join(path,f) for f in result if f.endswith('.jpg')]
test_im = np.array(cv2.imread(imlist[0],0))
H = test_im.shape[0]
W = test_im.shape[1]
comp = 10000
imgMatrix = np.zeros((H*W))



for i,f in enumerate(imlist[stf:enf]):
    im_c = cv2.imread(f)
    im_g = cv2.cvtColor(im_c,cv2.COLOR_RGB2GRAY)
    imgMatrix = np.vstack((imgMatrix,np.array(im_g).flatten()))


data = np.delete(imgMatrix, 0, axis=0)
pca = PCA(n_components = comp)
X_ = pca.fit_transform(data)
X = preprocessing.scale(X_)
print(X)
db = DBSCAN(eps = 6.327, min_samples=2).fit(X)
skl_labels = db.labels_
frm_index = np.arange(stf+1,enf+1,1)
print(np.vstack((frm_index,skl_labels)))