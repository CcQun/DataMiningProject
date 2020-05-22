import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

dataRoot = "D:\\CodingProject\\PyCharmProject\\DataMiningProject\\Data\\csv"

num_classes = 8
num_features = 408

datas = []
labels = []
for dataCsv in os.listdir(dataRoot):
    df = pd.read_csv(os.path.join(dataRoot, dataCsv))
    datas.append(df.iloc[:, :-1].values)
    labels.append(df.iloc[:, -1].values.reshape((-1, 1)))

data = np.vstack((datas[0], datas[1]))
label = np.vstack((labels[0], labels[1]))
for i in range(2, len(datas)):
    data = np.vstack((data, datas[i]))
    label = np.vstack((label, labels[i]))

label = np.eye(num_classes)[label.reshape(-1).astype(np.int8)]

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0, test_size=0.2)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')

x = tf.placeholder(tf.float32,shape=(None,num_features))