import Dataprocess.trans_Ori as tO
import numpy as np

np.set_printoptions(suppress=True)

def Qua2Eul(Qdata):
    return tO.Q2EA(Qdata)

def Qua2Dcm(Qdata):
    return tO.Q2DCM(Qdata)


#acb-ABC
# cluster-1
# 0.00703199588676
# -0.105938238486
# -2.92333794064
# cluster-2
# -0.000211690755445
# -0.0677324816844
# 2.99212767446

if __name__ == '__main__':
    Eul = np.array([-0.000211690755445,2.99212767446,-0.0677324816844])
    R = np.zeros((4,4))
    R[0,0] = np.cos(Eul[0])*np.cos(Eul[1])-np.sin(Eul[0])*np.sin(Eul[1])*np.sin(Eul[2])
    R[0,1] = np.sin(Eul[0])*np.cos(Eul[2])
    R[0,2] = np.cos(Eul[0])*np.sin(Eul[1])+np.sin(Eul[0])*np.cos(Eul[1])*np.sin(Eul[2])
    R[0,3] = 0
    R[1,0] = -np.sin(Eul[0])*np.cos(Eul[1])-np.cos(Eul[0])*np.sin(Eul[1])*np.sin(Eul[2])
    R[1,1] = np.cos(Eul[0])*np.cos(Eul[2])
    R[1,2] = -np.sin(Eul[0])*np.sin(Eul[1])+np.cos(Eul[0])*np.cos(Eul[1])*np.sin(Eul[2])
    R[1,3] = 0
    R[2,0] = -np.sin(Eul[1])*np.cos(Eul[2])
    R[2,1] = -np.sin(Eul[2])
    R[2,2] = np.cos(Eul[1])*np.cos(Eul[2])
    R[2,3] = 0
    R[3,0] = 0
    R[3,1] = 0
    R[3,2] = 0
    R[3,3] = 1

    print(R)