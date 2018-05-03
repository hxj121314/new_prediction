# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN,KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt

stf = 0
enf = 500
k = 50

path = 'D:/testimg'
result = os.listdir(path)
result.sort(key= lambda x:int(x[:-4]))
imlist = [os.path.join(path,f) for f in result if f.endswith('.jpg')]
test_im = np.array(cv2.imread(imlist[0],0))
H = 400
W = 800
# histMatrix = np.ones((256,1))
imgMatrix = np.zeros((H*W))

# im_c = cv2.imread(imlist[100],0)
# im_g = cv2.cvtColor(im_c,cv2.COLOR_BAYER_BG2BGR)
# hist = cv2.calcHist([im_g], [0], None, [256], [0.0,255.0])


for i,f in enumerate(imlist[stf:enf]):
    im_r = cv2.imread(f)
    im_c = cv2.resize(im_r,(H,W))
    im_g = cv2.cvtColor(im_c,cv2.COLOR_RGB2GRAY)
    # hist = cv2.calcHist([im_g], [0], None, [256], [0.0,255.0])
    # hist = hist/255
    # histMatrix = np.hstack((histMatrix, hist))
    imgMatrix = np.vstack((imgMatrix,np.array(im_g).flatten()))


data = np.delete(imgMatrix, 0, axis=0)
X = preprocessing.scale(data)
seed_ind = np.around(np.linspace(0,499,k)).astype(int)
print(seed_ind)
db = KMeans(n_clusters=k,init=X[seed_ind,:], n_init=1).fit(X)
skl_labels = db.labels_
frm_index = np.arange(stf+1,enf+1,1)
print(np.vstack((frm_index,skl_labels+((stf/500+1)*50))))
np.savetxt(path+'/result1.csv',np.vstack((frm_index,skl_labels+((stf/500+1)*50))).T)