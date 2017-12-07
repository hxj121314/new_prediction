#!/usr/bin/python
#coding:utf-8
import os
import tensorflow as tf
import math
import data_main as dm

import numpy as np
BATCH_SIZE = 10
cwd = './data/'
Fw_Interval = 35
Bk_Interval = 1


TFfile=cwd+"train.tfrecords"
signal=[]
labels2=[]
labels1=[]


writer = tf.python_io.TFRecordWriter(TFfile)  # to be generated TFrecord file
Rd_Data = dm.feedin([0],0)[:,2:6].transpose(1,0)
Train_Data = Rd_Data.astype(np.float32)
Glob_Step = Train_Data.shape[1] - Fw_Interval - Bk_Interval - 1
for Step in range(Glob_Step):
    da = Train_Data[: , Step:Step+Fw_Interval]
    la = Train_Data[: , Step+Fw_Interval+Bk_Interval]
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[la.tostring()])),
        'mat_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[da.tostring()]))
    }))
    writer.write(example.SerializeToString())  # write example to TFrecord file



# class_path = cwd +'train'+'/'
# for mat_name in os.listdir(class_path):
#     mat_file = class_path+ mat_name  # address of every matfile
#
#     y = scipy.io.loadmat(mat_file)
#     a=y['h']
#     a=np.reshape(a,[2,36])
#     da=a[:,0:32]
#     la=a[:,32:36]
#     print(da)
#     print(la)
#     example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[la.tostring()])),
#             'mat_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[da.tostring()]))
#         }))  # generate an example
#     writer.write(example.SerializeToString())  # write example to TFrecord file

writer.close()

