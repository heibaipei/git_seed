import tensorflow as tf
import scipy.io as sc
import numpy as np
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import xgboost as xgb

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

###extract the delta from the data
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def get_data():

	len_sample=1
	full=7000
	n_fea=14
	len_a=full/len_sample  # 6144 class1
	label0=np.zeros(len_a)
	label1=np.ones(len_a)
	label2=np.ones(len_a)*2
	label3=np.ones(len_a)*3
	label4=np.ones(len_a)*4
	label5=np.ones(len_a)*5
	label6=np.ones(len_a)*6
	label7=np.ones(len_a)*7
	label=np.hstack((label0,label1,label2,label3,label4,label5,label6,label7))
	label=np.transpose(label)
	label.shape=(len(label),1)
	print (label)

	

	time1 =time.clock()
	feature = sc.loadmat("EID-S.mat")  # EID-S, with 1 trial, 7000 samples per subject
	all = feature['eeg_close_8sub_1file']
	all = all[0:full*8, 0:n_fea]
	return label,all
