import Dataprocess.trans_Ori as tO
import numpy as np
import tensorflow as tf


def Qua2Eul(Qdata):
    return tO.Q2EA(Qdata)