from LSTM.loss_Function import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

LEARNINGRATING = 0.01
HIDDEN_SIZE = 512
NUM_LAYERS = 3
TIMESTEPS = 20
PREDICTSTEPS = 5
TRAINING_STEPS = 10000
BATCH_SIZE = 32
# TRAINING_EXAMPLES = 10000
# TESTING_EXAMPLES = 1000

#########################################################MY_LSTM
def lstm_cell():
    return tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True), output_keep_prob=0.5)



def lstm_model(X, y):
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(NUM_LAYERS)], state_is_tuple=True)
    x_ = X
    output, _ = tf.nn.dynamic_rnn(cell, x_, dtype=tf.float32)
    output = output[:,-1]
    prediction, _ = tf.contrib.learn.models.linear_regression(output, y)
    prediction = np.pi*tf.nn.tanh(prediction)
    loss = my_loss(prediction, y)


    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer = "Adagrad", learning_rate=LEARNINGRATING
    )
    return prediction, loss, train_op
