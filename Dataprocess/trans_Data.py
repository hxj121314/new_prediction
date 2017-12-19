import Dataprocess.trans_Ori as tO
import numpy as np

def Qua2Eul(Qdata):
    return tO.Q2EA(Qdata)

def Qua2Dcm(Qdata):
    return tO.Q2DCM(Qdata)