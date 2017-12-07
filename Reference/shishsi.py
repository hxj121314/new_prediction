import data_main as dm
import transtfrecords as ttf
import time
import tensorflow as tf
import lstm_model
import param
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
import os

# 每10秒加载一次最新的模型， 并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 2


MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "lstm_model.ckpt"



Rd_Data = dm.feedin([0] , 0)[:,2:6].transpose(1,0)
Test_Data = Rd_Data.astype(np.float32)
truv=Test_Data[:,ttf.Fw_Interval+ttf.Bk_Interval:-1]

matmat = tf.placeholder(tf.float32, [1, ttf.Fw_Interval, 4])
y = lstm_model.lstm(matmat, False, None)
saver = tf.train.Saver()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    saver.restore(sess, ckpt.model_checkpoint_path)
    pred = np.zeros((1,4),dtype='float')
    for Step in range(ttf.Glob_Step):
        da = Test_Data[:,Step:Step+ttf.Fw_Interval]
        da = np.reshape(da,(1,ttf.Fw_Interval,4))
        yy = sess.run(y, feed_dict={matmat: da})
        pred = np.vstack((pred, yy))
    pred = pred.transpose(1,0)
    coord.request_stop()
    coord.join(threads)
    plt.subplot(241)
    plt.plot(np.reshape(truv[0,:],[-1]))
    plt.subplot(245)
    plt.plot(np.reshape(pred[0,:],[-1]))
    plt.subplot(242)
    plt.plot(np.reshape(truv[1, :], [-1]))
    plt.subplot(246)
    plt.plot(np.reshape(pred[1, :], [-1]))
    plt.subplot(243)
    plt.plot(np.reshape(truv[2, :], [-1]))
    plt.subplot(247)
    plt.plot(np.reshape(pred[2, :], [-1]))
    plt.subplot(244)
    plt.plot(np.reshape(truv[3, :], [-1]))
    plt.subplot(248)
    plt.plot(np.reshape(pred[3, :], [-1]))

    plt.show()

