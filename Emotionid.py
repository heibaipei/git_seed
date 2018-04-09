import tensorflow as tf
import scipy.io as sc
import numpy as np
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import xgboost as xgb
import shutil
import os


def one_hot(y_):
    '''
    # this function is used to transfer one column label to one hot label
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    @return: one_hot_code
    '''
    y_ = y_.reshape(len(y_))
    y_ = y_.astype(int) + 1  # 因为onehot不能处理负数，所以把所有的数加上最小的负数的绝对值，即1
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    extract the delta from the data
    @return: b,a
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    @return: y
    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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


def get_train_test_datas(dataset_dir='../output/'):
    '''
    @param dataset_dir:the directory store the datas
    @return: feature_all:[data];all [data,label]
    @return: train_x,train_y,test_x,test_y
    '''
    import pickle

    input_height = input_width = 9
    window_size = 200
    lowcut = 12
    highcut = 30
    fs = 128
    ###############################################################################################
    # get train datasets
    ###############################################################################################
    with open(dataset_dir+"_data_200.pkl", "rb") as fp:
        feature_training = pickle.load(fp).reshape([-1, 16200])
        temp = []
        for i in range(16200):
            t = feature_training[:, i]
            t = butter_bandpass_filter(t, lowcut, highcut, fs, order=3)
            temp.append(t)
        feature_training = np.transpose(np.array(temp))
        feature_training = preprocessing.scale(
            feature_training).reshape([-1, 1, 16200])
    with open(dataset_dir+"_label_200.pkl", "rb") as fp:
        label_training = one_hot(pickle.load(fp))

    ###############################################################################################
    # get test datasets
    ###############################################################################################
    with open(dataset_dir+"_data_test_200.pkl", "rb") as fp:
        feature_testing = pickle.load(fp).reshape([-1, 16200])
        temp = []
        for i in range(16200):
            t = feature_testing[:, i]
            t = butter_bandpass_filter(t, lowcut, highcut, fs, order=3)
            temp.append(t)
        feature_testing = np.transpose(np.array(temp))
        feature_testing = preprocessing.scale(
            feature_testing).reshape([-1, 1, 16200])
    with open(dataset_dir+"_label_test_200.pkl", "rb") as fp:
        label_testing = one_hot(pickle.load(fp))

    return feature_training, label_training, feature_testing, label_testing


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (8 batch  7000 steps, 14 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X.shape is [ ?,14]

    # 3 hidden layer
    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])
    X_hidd2 = tf.sigmoid(
        tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2'])
    X_hidd3 = tf.sigmoid(
        tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3'])
    X_hidd4 = tf.sigmoid(
        tf.matmul(X_hidd3, weights['hidd4']) + biases['hidd4'])
    X_hidd5 = tf.sigmoid(
        tf.matmul(X_hidd4, weights['hidd5']) + biases['hidd5'])
    # X_in = tf.reshape(X_hidd1, [-1, n_steps, n_hidden2_units])
    # 注意啦，要把所有的特征放在LSTM啦，并且有可以不知道的分段在里边n_steps
    # ok
    ##########################################

    # 第四个 hidden layer basic LSTM Cell.
    '''
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(
        n_hidden1_units, forget_bias=1, state_is_tuple=True)
    ####定义初始状态#####
    init_state = lstm_cell_1.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        lstm_cell_1, X_in, initial_state=init_state, time_major=False)

    # outputs
    # final_states is the last outputs
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    # attention based model
    X_att2 = final_state[0]  # weights
    outputs_att = tf.multiply(outputs[-1], X_att2)
    '''
    results = tf.nn.softmax(tf.matmul(X_hidd5, weights['out']) + biases['out'])

    return results


##################################################################### 跑模型 ##################################################################
# tensorboard things
logfile = "log"
if os.path.exists(logfile):
    shutil.rmtree(logfile)

# 获取数据
feature_training, label_training, feature_testing, label_testing = get_train_test_datas()

# 定义一些东西
n_steps = 1
a_person_alldata_size = 7000
# 把它转化成一维的数组
feature_number = 16200
batch_size = 200

# batch split
n_group = feature_training.shape[0]//batch_size

train_fea = []
for i in range(n_group):
    f = feature_training[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)

train_label = []
for i in range(n_group):
    f = label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)

test_fea = []
for i in range(n_group//2):
    f = feature_testing[(0+batch_size*i):(batch_size+batch_size*i)]
    test_fea.append(f)

test_label = []
for i in range(n_group//2):
    f = label_testing[(0+batch_size*i):(batch_size+batch_size*i), :]
    test_label.append(f)


# 下面是和模型有关的
nodes = 8192
lameda = 0.001
lr = 0.01
train_times = 4000

# hyperparameters
n_inputs = feature_number  # the size of input layer
n_hidden1_units = 4096 * 2   # neurons in hidden layer
n_hidden2_units = 4096 // 2
n_hidden3_units = 4096 // 4
n_hidden4_units = 4096 // 16
n_hidden5_units = 4096 // 32
n_hidden6_units = 4096 // 32
n_classes = 3  # the size of output layer,there are 3 different kind of classes


# Define weights and biases
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
    'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),

    'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
    'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
    'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
    'hidd5': tf.Variable(tf.random_normal([n_hidden4_units, n_hidden5_units])),
    'hidd6': tf.Variable(tf.random_normal([n_hidden5_units, n_hidden6_units])),

    'out': tf.Variable(tf.random_normal([n_hidden6_units, n_classes]), trainable=True),
    'att': tf.Variable(tf.random_normal([n_inputs, n_hidden6_units]), trainable=True),
    'att2': tf.Variable(tf.random_normal([1, batch_size]), trainable=True),
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),

    'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units])),
    'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
    'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
    'hidd5': tf.Variable(tf.constant(0.1, shape=[n_hidden5_units])),
    'hidd6': tf.Variable(tf.constant(0.1, shape=[n_hidden6_units])),

    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
    'att': tf.Variable(tf.constant(0.1, shape=[n_hidden6_units])),
    'att2': tf.Variable(tf.constant(0.1, shape=[n_hidden6_units])),
}


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

pred = RNN(x, weights, biases)

# L2 loss prevents this overkill neural network to overfit the data
l2 = lameda * sum(tf.nn.l2_loss(tf_var)
                  for tf_var in tf.trainable_variables())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=pred, labels=y))+l2  # Softmax loss


train_op = tf.train.AdamOptimizer(lr).minimize(cost)
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
    for step in range(train_times):

        i = np.random.randint(0, n_group)
        ################ train #################
        # for i in range(n_group):
        _, train_accuracy, train_cost, summary = sess.run([train_op, accuracy, cost, train_merge], feed_dict={
            x: train_fea[i],
            y: train_label[i]
        })
        #summary = sess.run(merged)
        ############## record it to tensorboard #################
        train_writer.add_summary(summary, step)

        sess.run(tf.local_variables_initializer())
        test_accuracy_average = 0.0
        for i in range(n_group//2):
            test_accuracy, streaming_loss, streaming_accuracy = sess.run([accuracy, streaming_loss_update, streaming_accuracy_update],
                                                                         feed_dict={x: test_fea[i],
                                                                                    y: test_label[i]
                                                                                    })
            test_accuracy_average += test_accuracy
        test_accuracy_average /= float(n_group//2)
        summary = sess.run(test_merge)
        test_writer.add_summary(summary, step)

        ############### early stopping ##################
        if test_accuracy > 0.999:
            print(
                "The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ", test_accuracy)
            break

        ############### print something ##################
        if step % 10 == 0:
            #    pp = sess.run(pred_result, feed_dict={x: test_fea[i//2],
            #                                          y: test_label[i//2]})
            # i_want_to_see("predict", pp[0:10])
            #    gt = np.argmax(label_testing, 1)
            # i_want_to_see("groundtruth", gt[0:10])
            #    hh = sess.run(accuracy, feed_dict={x: test_fea[i//2],
            #                                       y: test_label[i//2]}
            #                  )
            #    h2 = sess.run(accuracy,  feed_dict={x: train_fea[i],
            #                                        y: train_label[i]})
            # i_want_to_see("training acc", h2)
            print("The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step,
                  ", The test accuracy is:", test_accuracy_average, ", The train accuracy is:", train_accuracy)
            print("The cost is :", train_cost)

    train_writer.close()
    test_writer.close()

####下面的以后再管############
#     B = sess.run(Feature, feed_dict={
#         x: train_fea[0],
#         y: train_label[0],
#     })
#     for i in range(n_group):
#         D = sess.run(Feature, feed_dict={
#             x: train_fea[i],
#             y: train_label[i],
#         })
#         B = np.vstack((B, D))
#     B = np.array(B)
#     Data_train = B  # Extracted deep features
#     Data_test = sess.run(Feature, feed_dict={x: test_fea[i//2],
#                                              y: test_label[i//2]})

# ########### 下面这些东西看不懂，以后再研究 ###########
# # XGBoost
# xgb_s = time.clock()
# xg_train = xgb.DMatrix(Data_train, label=np.argmax(label_training, 1))
# xg_test = xgb.DMatrix(Data_test, label=np.argmax(label_testing, 1))

# # setup parameters for xgboost
# param = {}
# # use softmax multi-class classification
# param['objective'] = 'multi:softprob'  # can I replace softmax by SVM??
# # softprob produce a matrix with probability value of each class
# # scale weight of positive examples
# param['eta'] = 0.7

# param['max_depth'] = 6
# param['silent'] = 1
# param['nthread'] = 4
# param['subsample'] = 0.9
# param['num_class'] = n_classes

# np.set_printoptions(threshold=np.nan)
# watchlist = [(xg_train, 'train'), (xg_test, 'test')]
# num_round = 500
# bst = xgb.train(param, xg_train, num_round, watchlist)
# time8 = time.clock()
# pred = bst.predict(xg_test)
# xgb_e = time.clock()
# print('xgb run time', xgb_e - xgb_s)
# print('RNN acc', test_accuracy_average)
