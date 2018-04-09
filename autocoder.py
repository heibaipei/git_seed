'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np 
import tensorflow as tf
from keras.layers.convolutional import UpSampling2D
from keras import optimizers
import math
import os
os.sys.path.append('../')
import dataset.DataSet as DB
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

dropout = 0.5

infodata = DB.Get5Class(ratio = 0.7, tag_array = [0, 0, 1, 1, 1], label_arry=[0, 0, 0, 0, 1], nlabel=0)
x_scale_train, x_scale_test, _, _ = infodata.GetScaleData()

N = infodata.GetSizeTrain()
D = infodata.GetDim()
num_classes = infodata.GetNumClass()

num_fc_1 = 500
learning_rate = 1e-4  
batch_size = 15

training_iters = 2100000
display_step = 14

# tf Graph input
x = tf.placeholder(tf.float32, [None, D])
#y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

def conv2d(x, w_shape, w_name, b_shape, b_name):
    # Conv2D wrapper, with bias and relu activation
    outshape = tf.shape(x)
    filter_w = weight_variable(w_shape, w_name)
    x = tf.nn.conv2d(x, filter_w, strides=[1, 1, 1, 1], padding='SAME')
    b = bias_variable(b_shape, b_name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x), filter_w, outshape

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, 1, 1], strides=[1, k, 1, 1], padding='SAME')

def uppool2d(x, upshape):
    #out = tf.concat(2, [x, tf.zeros_like(x)])
    #out = tf.concat(2, [x, x])
    #out = tf.concat([x, tf.zeros_like(x)], 2)
    out = tf.concat([x, x], 2)
    out = tf.reshape(out, upshape)
    return out

def full_layer(x, w_shape, w_name, b_shape, b_name):
    w_fc1 = weight_variable(w_shape, w_name)
    b_fc1 = bias_variable(b_shape, b_name)    
    fc1 = tf.reshape(x, [-1, w_shape[0]])
    fc1 = tf.add(tf.matmul(fc1, w_fc1), b_fc1)
    fc1 = tf.nn.relu(fc1)
    return tf.nn.dropout(fc1, dropout), w_fc1

def out_layer(x, w_shape, w_name, b_shape, b_name):
    w_out = weight_variable(w_shape, w_name)
    b_out = bias_variable(b_shape, b_name)  
    return tf.add(tf.matmul(x, w_out), b_out), w_out
# Create model

x_input = tf.reshape(x, [-1,D,1,1])
# conv1
conv1, wc1, outshape1 = conv2d(x_input, [1, 2, 1, 16], 'wc1',  [16], 'bc1')
poolshape1 = tf.shape(conv1)
conv1_p = maxpool2d(conv1, k=2)
# conv2
conv2, wc2, outshape2 = conv2d(conv1_p, [1, 2, 16, 8], 'wc2',  [8], 'bc2')
poolshape2 = tf.shape(conv2)
conv2_p = maxpool2d(conv2, k=2)

# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
width_pool = int(np.floor(D/4))
fc_num_inputs = width_pool*8
fc1_shape = [fc_num_inputs, num_fc_1]
coder, wfc1 = full_layer(conv2_p, fc1_shape, 'wfc1', [num_fc_1], 'bfc1')
print ("coder", coder)
  
#decoder
de_bfc1 =  bias_variable([fc_num_inputs], 'de_bfc1')
de_wfc1 = tf.transpose(wfc1)
de_fc1  = tf.add(tf.matmul(coder, de_wfc1), de_bfc1)
de_fc1 = tf.nn.relu(de_fc1)
de_fc1 = tf.reshape(de_fc1, [-1, width_pool, 1, 8])
print ("de_fc1", de_fc1)
#de_conv2_p = UpSampling2D((2, 1))(de_fc1)
de_conv2_p = uppool2d(de_fc1, poolshape2)
print ("de_conv2_p", de_conv2_p.get_shape())
de_bc2 = bias_variable([wc2.get_shape().as_list()[2]], "de_bc2")
print ("de_bc2", de_bc2)
print ("wc2", wc2)
print ("outshape2", outshape2)
de_conv2 = tf.nn.relu(tf.add(tf.nn.conv2d_transpose(de_conv2_p, wc2, outshape2,  strides=[1, 1, 1, 1], padding='SAME'), de_bc2))
print ("de_conv2", de_conv2.get_shape())
#de_conv2_p = UpSampling2D((2, 1))(de_conv2)
de_conv2_p = uppool2d(de_conv2, poolshape1)

de_bc1 = bias_variable([wc1.get_shape().as_list()[2]], 'de_bfc1')
de_conv1 = tf.add(tf.nn.conv2d_transpose(de_conv2_p, wc1, outshape1,  strides=[1, 1, 1, 1], padding='SAME'), de_bc1)
de_conv1 = tf.nn.relu(de_conv1) 
de_out = de_conv1

# Define loss and optimizer
cost = tf.reduce_sum(tf.square(de_out - x_input))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_ind = np.random.choice(N,batch_size,replace=False)
        batch_x = x_scale_train[batch_ind]
#        batch_y = infodata.y_train[batch_ind]
#         temp1, temp2 ,temp3, temp4 = sess.run([x_input, de_out, cost, de_conv2_p] , feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
#         print (np.shape(temp1))
#         print (np.shape(temp2))
#         print (np.shape(temp4))
#         print (temp4[0, 0:4, 0, :])
#         print (sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout}))
#         break
        sess.run(optimizer, feed_dict={x: batch_x, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, keep_prob: 1.})
            print ("{:.5f}".format(loss) )
        step += 1
    print("Optimization Finished!")

