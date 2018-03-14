import Dataprocess.trans_Data as tD
import Dataprocess.get_Data as gD
import numpy as np

from Dataprocess.trans_Data import *
from Dataprocess.get_Data import *

# ind = [2,3,4,5]
# Qda = gD.getData(11, 0)[:, ind]
# Dda = tD.Qua2Dcm(Qda)
# print(Dda[:,:,4])
# Dda2 = Dda.reshape(16,-1)
# print(Dda2[:,4])
# np.savetxt('D:\DCMData/mdata.csv', Dda2.transpose(1,0), delimiter = ',')



# ind = [2,3,4,5]
# test_Data = Qua2Eul(getData(11, 0)[:, ind]).transpose(1, 0)
# print(test_Data[:,1])

ind = [2,3,4,5]
for i in range(50):
    if i == 0:
        test_Data = Qua2Eul(getData(i, 0)[0:10, ind])
    else:
        test_Data_new = Qua2Eul(getData(i, 0)[0:10, ind])
        test_Data = np.vstack((test_Data,test_Data_new))

print(test_Data)