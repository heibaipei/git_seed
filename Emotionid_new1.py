import tensorflow as tf
import scipy.io as sc
import numpy as np
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import xgboost as xgb
import shutil
import os
from data_getter import get_train_datas, get_test_datas


def i_want_to_see(name, content):
    '''
    print what you want to see in a cool way
    @param name:
    @param content:
    '''

    print('#####################################################################################')
    print(name)
    print('-------------------------------------------------------------------------------------')
    print(content)
    print('#####################################################################################')


def RNN(X):

    # neurons in hidden layer
    n_hidden1_units = 128
    n_hidden2_units = 64
    n_hidden3_units = 64
    n_hidden4_units = 64
    n_hidden5_units = 64
    n_hidden6_units = 128

    # Define weights and biases
    weights = {
        'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
        'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
        'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
        'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
        'hidd5': tf.Variable(tf.random_normal([n_hidden4_units, n_hidden5_units])),
        'hidd6': tf.Variable(tf.random_normal([n_hidden5_units, n_hidden6_units])),
        'out': tf.Variable(tf.random_normal([n_hidden6_units, n_classes]), trainable=True),
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
        'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units])),
        'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
        'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
        'hidd5': tf.Variable(tf.constant(0.1, shape=[n_hidden5_units])),
        'hidd6': tf.Variable(tf.constant(0.1, shape=[n_hidden6_units])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
    }

    # hidden layer for input to cell
    ########################################

    # X(batch_size,feature_numbers)
    X = tf.reshape(X, [-1, feature_number])

    # 6 hidden layer
    X_hidd1 = tf.nn.relu(
        tf.add(
            tf.matmul(X, weights['in']),
            biases['in']))
    X_hidd2 = tf.nn.relu(
        tf.add(
             tf.matmul(X_hidd1, weights['hidd2']),
             biases['hidd2']))
     X_hidd3 = tf.nn.relu(
        tf.add(
             tf.matmul(X_hidd2, weights['hidd3']),
             biases['hidd3']))
    X_hidd4 = tf.nn.relu(
        tf.add(
            tf.matmul(X_hidd3, weights['hidd4']),
            biases['hidd4']))
    X_hidd5 = tf.nn.relu(
        tf.add(
            tf.matmul(X_hidd4, weights['hidd5']),
            biases['hidd5']))
    X_hidd6 = tf.nn.relu(
        tf.add(
            tf.matmul(X_hidd5, weights['hidd6']),
            biases['hidd6']))

    # X_hidd6(?,n_hidden6_units=256)
    # ====>
    # X_in(batch_size=?,n_steps=16,n_hidden6_units=256)
    X_in = tf.reshape(X_hidd6, [-1, n_steps, n_hidden6_units])

    # hidden layer basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        n_hidden1_units, forget_bias=1, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        lstm_cell, X_in, initial_state=init_state, time_major=False)

    # outputs
    # final_states is the last outputs
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    #results = tf.matmul(outputs[-1], weights['out']) + \
        #biases['out']  # 选取最后一个 output

    # # attention based model
    X_att2 = final_state[0]  # weights
    i_want_to_see(X_att2, X_att2)
    outputs_att = tf.multiply(outputs[-1], X_att2)
    results = outputs_att

    return results  # outputs_att # 


##################################################################### 跑模型 ##################################################################
# tensorboard things
logfile = "log"
if os.path.exists(logfile):
    shutil.rmtree(logfile)

# 定义一些东西
n_steps = 1

n_inputs = feature_number = 62*2  # the size of input layer
batch_size = 4096


# 下面是和模型有关的
lameda = 0.001
train_times = 50000

learning_rate = 0.001
# tf.train.exponential_decay(
#     learning_rate=0.01,
#     global_step=train_times,
#     decay_steps=100,
#     decay_rate=0.96,
#     staircase=True,
#     name=None
# )


n_classes = 3  # the size of output layer,there are 3 different kind of classes


# tf Graph input
x = tf.placeholder(tf.float32, [None, feature_number], name="features")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

pred = RNN(x)

# L2 loss prevents this overkill neural network to overfit the data
l2 = lameda * sum(tf.nn.l2_loss(tf_var)
                  for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y))+l2  # Softmax loss


train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
pred_result = tf.argmax(pred, 1, name="pred_result")
label_true = tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

summary_loss = tf.summary.scalar("loss", cost)
summary_accuracy = tf.summary.scalar("accuracy", accuracy)

streaming_loss, streaming_loss_update = tf.contrib.metrics.streaming_mean(cost)
streaming_loss_scalar = tf.summary.scalar('loss', streaming_loss)

streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(
    accuracy)
streaming_accuracy_scalar = tf.summary.scalar('accuracy', streaming_accuracy)

train_merge = tf.summary.merge([summary_loss, summary_accuracy])
test_merge = tf.summary.merge(
    [streaming_loss_scalar, streaming_accuracy_scalar])


init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
rnn_s = time.clock()
with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logfile+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(logfile+"/test", sess.graph)

    feature_training, label_training = get_train_datas()
    feature_testing, label_testing = get_test_datas()

    for step in range(train_times):

        lr = learning_rate  # sess.run(learning_rate)

        ################ train #################

        batch_start_index = np.random.randint(
            feature_training.shape[0]-batch_size-1)
        batch_x = feature_training[batch_start_index:batch_start_index+batch_size]
        batch_y = label_training[batch_start_index:batch_start_index+batch_size]

        _, train_accuracy, train_cost, summary = sess.run([train_op, accuracy, cost, train_merge], feed_dict={
            x: batch_x,
            y: batch_y,
        })

        ############## record it to tensorboard #################
        train_writer.add_summary(summary, step)

        sess.run(tf.local_variables_initializer())

        # test_accuracy, streaming_loss, streaming_accuracy = sess.run([accuracy, streaming_loss_update, streaming_accuracy_update],
        #                                                              feed_dict={x: feature_testing,
        #                                                                         y: label_testing,
        #                                                                         })

        # summary = sess.run(test_merge)
        # test_writer.add_summary(summary, step)

        # ############### print something ##################
        # if step % 50 == 0:

        #     feature_training, label_training = get_train_datas()

        #     print("The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step,
        #           ", The test accuracy is:", test_accuracy, ", The train accuracy is:", train_accuracy)
        #     print("The cost is :", train_cost)

        ############### early stopping ##################
        # if test_accuracy > 0.9999:
        #     print(
        #         "The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ", test_accuracy)
        #     break

    train_writer.close()
    test_writer.close()
