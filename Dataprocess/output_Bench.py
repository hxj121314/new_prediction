import Dataprocess.trans_Data as tD
import Dataprocess.get_Data as gD
import numpy as np

ind = [2,3,4,5]
Qda = gD.getData(11, 0)[:, ind]
Dda = tD.Qua2Dcm(Qda)
print(Dda[:,:,4])
Dda2 = Dda.reshape(16,-1)
print(Dda2[:,4])
np.savetxt('D:\DCMData/mdata.csv', Dda2, delimiter = ',')