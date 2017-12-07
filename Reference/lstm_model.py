import tensorflow as tf
import param
import numpy as np

learn = tf.contrib.learn

def lstm(input_tensor, train, regularizer):
    # 注意类型必须为 tf.float32

    # 每个隐含层的节点数
    hidden_size = 512
    # LSTM layer 的层数
    layer_num = 3
    # 最后输出分类类别数量，如果是回归预测的话应该是 1
    FC_SIZE = 32
    OUTPUT_NODE = 4
    batch_size = param.BATCH_SIZE
    # batch_size = 1
    keep_prob = 0.5

    def lstm_cell():

        return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True),output_keep_prob=keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

    # **步骤5：用全零来初始化state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
    # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
    # ** state.shape = [layer_num, 2, batch_size, hidden_size],
    # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
    # ** 最后输出维度是 [batch_size, hidden_size]
    outputs, state = tf.nn.dynamic_rnn(cell, inputs=input_tensor, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
    # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
    # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
    # **步骤6：方法二，按时间步展开计算
    # outputs = list()
    # state = init_state
    # with tf.variable_scope('RNN'):
    #     for timestep in range(timestep_size):
    #         if timestep > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         # 这里的state保存了每一层 LSTM 的状态
    #         (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
    #         outputs.append(cell_output)
    # h_state = outputs[-1]
    # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    # out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
    # out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
    # 开始训练和测试

    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable("weight", [hidden_size, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1),dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(h_state, fc1_weights)+fc1_biases)#10，64
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases  # 10，64

    return  fc2