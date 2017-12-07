import tensorflow as tf
import os
import lstm_model
import param
import transtfrecords as ttf
from matplotlib import pyplot as plt

learn = tf.contrib.learn

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "lstm_model.ckpt"

def Train(TFfile):
    filename_queue = tf.train.string_input_producer([TFfile])  # read into stream，生成文件名队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'mat_data': tf.FixedLenFeature([], tf.string),
                                       })
    mat_data, label = features['mat_data'], features['label']
    mat_batch, label_batch = tf.train.shuffle_batch(  # 随机输出batch
        [mat_data, label], batch_size=param.BATCH_SIZE,
        capacity=2100, min_after_dequeue=2000)
    mat_batch = tf.decode_raw(mat_batch, tf.float32)
    label_batch = tf.decode_raw(label_batch, tf.float32)
    mat_batch = tf.reshape(mat_batch, [param.BATCH_SIZE, 4, ttf.Fw_Interval])
    mat_batch = tf.transpose(mat_batch, [0, 2, 1])
    regularizer = tf.contrib.layers.l2_regularizer(param.REGULARAZTION_RATE)

    output = lstm_model.lstm(mat_batch, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(param.MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    mean = tf.square(output-label_batch)
    loss = tf.reduce_mean(mean)+tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(param.LEARNING_RATE_BASE, global_step, 2000/ param.BATCH_SIZE,
                                               param.LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        init_o = tf.local_variables_initializer().run()
        sess.run(init_op, init_o)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        Loss = []
        for i in range(10000):
            # 验证和测试的过程将会有一个独立的程序来完成
            # 类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
            # 每100轮保存一次模型。
            value = sess.run(mean)
            _, loss_value, step = sess.run([train_op, loss, global_step])
            # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
            # 在验证数据集上的正确率信息会有一个单独的程序来生成。
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %f." % (step, loss_value))

                Loss.append(loss_value)
                # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

        coord.request_stop()
        coord.join(threads)
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        plt.plot(Loss)
        ax1.set_xlabel('step')
        ax1.set_ylabel('Cost')
        plt.title('Cross Loss')
        plt.grid()
        plt.show()

def main(argv=None):
    cwd = './data/'
    TFfile = cwd + "train.tfrecords"
    Train(TFfile)

if __name__ == '__main__':
    tf.app.run()