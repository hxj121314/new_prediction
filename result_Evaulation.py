import numpy as np

ANGLETHRESH = 5

def dif_Ang(prediction, label):
    differ = abs(prediction - label)
    ang_Thr = np.pi*(ANGLETHRESH/180.0)
    acu_Rate = (differ[0,:] < ang_Thr) & (differ[1,:] < ang_Thr) & (differ[2,:] < ang_Thr)
    acu_Rate = np.sum(acu_Rate)/len(acu_Rate)
    mean_Error = np.zeros([3,1],dtype=np.float)
    mean_Error[0] = differ[0,:].mean()
    mean_Error[1] = differ[1,:].mean()
    mean_Error[2] = differ[2,:].mean()
    return acu_Rate,mean_Error