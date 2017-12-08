import tensorflow as tf
import numpy as np
import new_LSTM.param as param

learn =  tf.contrib.learn

def lstm_cell():
    hidden_size = param.HIDDEN_SIZE
    keep_prob = param.KEEP_PROB
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True), output_keep_prob=keep_prob)

def lstm(input_tensor, train, regularizer):
    hidden_size = param.HIDDEN_SIZE
    layer_num = param.LAYER_NUM
    fc_size = param.FC_SIZE
    output_node = param.OUTPUT_NODE
    batch_size = param.BATCH_SIZE
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs=input_tensor, initial_state=init_state, time_major=False)
    h_state = outputs[:, -1, :]

    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable("weight", [hidden_size, fc_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [fc_size], initializer=tf.constant_initializer(0.1),dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(h_state, fc1_weights)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable("weight", [fc_size, output_node],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [output_node], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc2 = tf.nn.tanh(tf.matmul(fc1, fc2_weights) + fc2_biases)

    return fc2
