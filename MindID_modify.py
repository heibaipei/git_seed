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
    y_ = y_.astype(int)
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


def get_train_test_datas():
    '''
    @return: feature_all:[data];all [data,label]
    @return: train_x,train_y,test_x,test_y
    '''

    ###############################################################################################
    # get features
    ###############################################################################################
    # get x from EID-S.mat : EID-S, with 1 trial, 7000 samples per subject
    datas = sc.loadmat("EID-S.mat")
    x_all = datas['eeg_close_8sub_1file'][0:a_person_alldata_size *
                                          8, 0:feature_number]

    # filte x : EEG Delta pattern decomposition
    x_f = []
    fs = 128.0
    lowcut = 0.5
    highcut = 4.0
    for i in range(x_all.shape[1]):
        t = x_all[:, i]
        t = butter_bandpass_filter(t, lowcut, highcut, fs, order=3)
        x_f.append(t)
    x_all = np.transpose(np.array(x_f))

    # minus Direct Current(减去电流噪声)
    x_all = x_all-4200

    # z-score scaling
    # min-max  unity scaling
    # feature_all=preprocessing.minmax_scale(feature_all,feature_range=(0,1))
    # feature_all=feature_all/sum(feature_all)
    x_all = preprocessing.scale(x_all)

    ###############################################################################################
    # get labels
    ###############################################################################################
    label0 = np.zeros(a_person_a_data_size)
    label1 = np.ones(a_person_a_data_size)
    label2 = np.ones(a_person_a_data_size)*2
    label3 = np.ones(a_person_a_data_size)*3
    label4 = np.ones(a_person_a_data_size)*4
    label5 = np.ones(a_person_a_data_size)*5
    label6 = np.ones(a_person_a_data_size)*6
    label7 = np.ones(a_person_a_data_size)*7
    label = np.transpose(np.hstack((label0, label1, label2, label3,
                                    label4, label5, label6, label7)))
    # print ('label.shape',label.shape)
    label.shape = (len(label), 1)

    ###############################################################################################
    # combine features and labels
    ###############################################################################################
    x_and_y = np.hstack((x_all, label))

    ###############################################################################################
    # split train data and test data
    ###############################################################################################

    data_size = x_and_y.shape[0]

    # use the first subject as testing subject
    np.random.shuffle(x_and_y)

    train_data = x_and_y[0:(int(data_size*0.875))]  # 1 million samples
    test_data = x_and_y[(int(data_size*0.875)):]

    feature_training = train_data[:, 0:feature_number].reshape(
        [train_data.shape[0], n_steps, feature_number])
    label_training = one_hot(train_data[:, feature_number])

    feature_testing = test_data[:, 0:feature_number].reshape(
        [test_data.shape[0], n_steps, feature_number])
    label_testing = one_hot(test_data[:, feature_number])

    return feature_training, label_training, feature_testing, label_testing, data_size


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (8 batch  7000 steps, 14 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X.shape is [ ?,14]

    # 3 hidden layer
    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])
    X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
    X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
    X_in = tf.reshape(X_hidd3, [-1, n_steps, n_hidden4_units])
    # 注意啦，要把所有的特征放在LSTM啦，并且有可以不知道的分段在里边n_steps
    # ok
    ##########################################

    # 第四个 hidden layer basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(
        n_hidden3_units, forget_bias=1, state_is_tuple=True)
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
    results = tf.matmul(outputs_att, weights['out']) + biases['out']

    return (results, outputs[-1])


##################################################################### 跑模型 ##################################################################
# tensorboard things
logfile = "log"
if os.path.exists(logfile):
    shutil.rmtree(logfile)

# 定义一些东西
n_steps = 1
a_person_alldata_size = 7000
a_person_a_data_size = int(a_person_alldata_size/n_steps)
feature_number = 14

# 获取数据
feature_training, label_training, feature_testing, label_testing, data_size = get_train_test_datas()

# data_size= 7000*8 batch_size = 7000
batch_size = int(data_size*0.125)  # 注意这里的data_size 竟然是所有的数据的大小

# 数据分片
# batch split
n_group = 7
train_fea = []
for i in range(n_group):
    f = feature_training[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
    # train_fea.length 7000*8

train_label = []
for i in range(n_group):
    f = label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
    # train_label.length 7000*8


# 下面是和模型有关的
nodes = 30
lameda = 0.001
lr = 0.001
train_times = 2000

# hyperparameters
n_inputs = feature_number  # the size of input layer
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units = nodes
n_classes = 8  # the size of output layer,there are 8 different kind of classes


# Define weights and biases
weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
    'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),

    'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
    'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
    'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),

    'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
    'att': tf.Variable(tf.random_normal([n_inputs, n_hidden4_units]), trainable=True),
    'att2': tf.Variable(tf.random_normal([1, batch_size]), trainable=True),
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),

    'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units])),
    'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
    'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),

    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), trainable=True),
    'att': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
    'att2': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
}


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
y = tf.placeholder(tf.float32, [None, n_classes], name="labels")

pred, Feature = RNN(x, weights, biases)

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

tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
rnn_s = time.clock()
with tf.Session(config=config) as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(logfile+"/train", sess.graph)
    test_writer = tf.summary.FileWriter(logfile+"/test", sess.graph)
    for step in range(train_times):

        ################ train #################
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],  # 输入x第一批 7000*
                y: train_label[i]
            })

        ############## record it to tensorboard #################
        summary = sess.run(merged,  feed_dict={x: train_fea[i],
                                               y: train_label[i]})
        train_writer.add_summary(summary, step)

        test_accuracy, summary = sess.run([accuracy, merged], feed_dict={
            x: feature_testing, y: label_testing})
        test_writer.add_summary(summary, step)

        ############### early stopping ##################
        if test_accuracy > 0.999:
            print(
                "The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ", test_accuracy)
            break

        ############### print something ##################
        if step % 10 == 0:
            pp = sess.run(pred_result, feed_dict={
                          x: feature_testing, y: label_testing})
            # i_want_to_see("predict", pp[0:10])
            gt = np.argmax(label_testing, 1)
            i_want_to_see("groundtruth", gt[0:10])
            hh = sess.run(accuracy, feed_dict={
                x: feature_testing,
                y: label_testing,
            })
            h2 = sess.run(accuracy,  feed_dict={x: train_fea[i],
                                                y: train_label[i]})
            # i_want_to_see("training acc", h2)
            print("The lamda is :", lameda, ", Learning rate:", lr, ", The step is:", step,
                  ", The accuracy is:", hh, ", The train accuracy is:", h2)
            print("The cost is :", sess.run(cost, feed_dict={
                x: feature_testing,
                y: label_testing,
            }))

    train_writer.close()
    test_writer.close()

    B = sess.run(Feature, feed_dict={
        x: train_fea[0],
        y: train_label[0],
    })
    for i in range(1, n_group):
        D = sess.run(Feature, feed_dict={
            x: train_fea[i],
            y: train_label[i],
        })
        B = np.vstack((B, D))
    B = np.array(B)
    Data_train = B  # Extracted deep features
    Data_test = sess.run(Feature, feed_dict={
                         x: feature_testing, y: label_testing})

########### 下面这些东西看不懂，以后再研究 ###########
# XGBoost
xgb_s = time.clock()
xg_train = xgb.DMatrix(Data_train, label=np.argmax(label_training, 1))
xg_test = xgb.DMatrix(Data_test, label=np.argmax(label_testing, 1))

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'  # can I replace softmax by SVM??
# softprob produce a matrix with probability value of each class
# scale weight of positive examples
param['eta'] = 0.7

param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['subsample'] = 0.9
param['num_class'] = n_classes

np.set_printoptions(threshold=np.nan)
watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist)
time8 = time.clock()
pred = bst.predict(xg_test)
xgb_e = time.clock()
print('xgb run time', xgb_e - xgb_s)
print('RNN acc', hh)
