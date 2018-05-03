# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


stf = 0
enf = 200

path = 'D:/testimg'
result = os.listdir(path)
result.sort(key= lambda x:int(x[:-4]))
imlist = [os.path.join(path,f) for f in result if f.endswith('.jpg')]
test_im = np.array(cv2.imread(imlist[0],0))
histMatrix = np.ones((256,1))

# for i,f in enumerate(imlist[stf:enf]):
#     im_c = cv2.imread(f)
#     im_g = cv2.cvtColor(im_c,cv2.COLOR_RGB2GRAY)
#     hist = cv2.calcHist([im_g], [0], None, [256], [0.0,255.0])
#     plt.subplot(5,2,i+1)
#     plt.plot(hist)
#     plt.title(i+stf)
# plt.show()


for i,f in enumerate(imlist[stf:enf]):
    im_c = cv2.imread(f)
    im_g = cv2.cvtColor(im_c,cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([im_g], [0], None, [256], [0.0,255.0])
    hist = hist/255
    histMatrix = np.hstack((histMatrix,hist))

#225
data = np.delete(histMatrix, 0, axis=1)
X = preprocessing.scale(data.transpose())
# X = data.transpose()
db = DBSCAN(eps= 17.5, min_samples=2).fit(X)
skl_labels = db.labels_
frm_index = np.arange(stf+1,enf+1,1)
print(np.vstack((frm_index,skl_labels)).transpose())
print(skl_labels[np.argmax(skl_labels)])