'''
Created on 2018年4月5日

@author: Jack
'''
import os
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from scipy.signal import butter, lfilter

train_dataset_dir = '../seed_data/train/'
test_dataset_dir = '../seed_data/test/'
labels = [1, 0, - 1, - 1, 0, 1, - 1, 0, 1, 1, 0, - 1, 0, 1, - 1]
feature_number = 62


def one_hot(y_, values=[]):
    '''
    # this function is used to transfer one column label to one hot label
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    @param values:different values
    @return: one_hot_code
    '''
    y_ = y_.reshape(len(y_)).astype(int)
    n_values = len(values)

    # 因为onehot不能处理负数，所以把所有的数加上最小的负数的绝对值
    if min(values) < 0:
        y_ = y_-min(values)

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


def data_preprocess(datas):
    '''
    @return processed data
    '''
    # mess up the datas
    datas = np.array(datas).reshape(-1, feature_number+1)
    np.random.shuffle(datas)

    #这里会出错
    # filter the wave
    temp = np.array([])
    for i in range(datas.shape[1]-1):
        np.hstack((temp,
                   butter_bandpass_filter(
                       data=datas[:, i], lowcut=4, highcut=30, fs=200, order=3)))

    # 归一化处理
    datas = preprocessing.scale(datas)

    # split x and y
    x = datas[:, :feature_number]
    y = one_hot(datas[:, feature_number:], values=[-1, 0, 1])

    return x, y


def get_train_datas():
    '''
    get random datas from train dataset
    @return: train_x,train_y
    '''

    datas = []

    record_list = [task for task in os.listdir(
        train_dataset_dir) if os.path.isfile(os.path.join(train_dataset_dir, task))]
    record_name = record_list[np.random.randint(0, len(record_list))]

    record = sio.loadmat(train_dataset_dir+"/"+record_name)
    data_keys = [key for key in record.keys() if '1' in key]
    for eeg_num in data_keys:
        student_data = record[eeg_num].transpose(1, 0)
        y = labels[int(eeg_num) - 101]
        for line_num in range(len(student_data)-1):
            data = student_data[line_num].tolist()
            data.extend(student_data[line_num+1].tolist())
            data.append(y)
            datas.append(data)

    train_x, train_y = data_preprocess(datas)
    return train_x, train_y


def get_test_datas():
    '''
    get all datas from test dataset
    @return: test_x,test_y
    '''

    datas = []

    record_list = [task for task in os.listdir(
        test_dataset_dir) if os.path.isfile(os.path.join(test_dataset_dir, task))]
    for record in record_list:
        data_list = sio.loadmat(test_dataset_dir+"/"+record)
        for eeg_num in data_list.keys():
            if '1' in eeg_num:
                student_data = data_list[str(int(eeg_num))].transpose(1, 0)
                y = labels[int(eeg_num) - 101]
                for line_num in range(len(student_data)-1):
                    data = student_data[line_num].tolist()
                    data.extend(student_data[line_num+1].tolist())
                    data.append(y)
                    datas.append(data)

    test_x, test_y = data_preprocess(datas)

    return test_x, test_y
