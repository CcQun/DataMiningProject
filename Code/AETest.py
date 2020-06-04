import os
import numpy as np
import tensorflow as tf
# 为显示中文，导入中文字符集
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf')
import pandas as pd

dataRoot = "D:\\CodingProject\\PyCharmProject\\DataMiningProject\\"


class Autoencoder(object):
    def __init__(self, n_hidden_1, n_hidden_2, n_input, learning_rate, scale=0.1):
        self.training_scale = scale

        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_input = n_input

        self.learning_rate = learning_rate

        self.weights, self.biases = self._initialize_weights()

        self.x = tf.placeholder("float", [None, self.n_input])

        self.encoder_op = self.encoder(self.x)
        self.decoder_op = self.decoder(self.encoder_op)

        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
        }
        return weights, biases

    def encoder(self, X):
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(X + self.training_scale * tf.random_normal((self.n_input,), seed=0),
                             self.weights['encoder_h1']),
                   self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
        return layer_2

    def decoder(self, X):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h1']), self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
        return layer_2

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def transform(self, X):
        return self.sess.run(self.encoder_op, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoder_op, feed_dict={self.x: X})

    def save(self, model_path):
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)

    def restore(self, model_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)


save_path = dataRoot + 'model\\AE\\'
model = Autoencoder(n_hidden_1=210, n_hidden_2=60, n_input=408, learning_rate=0.01)
model.restore(save_path + "ae.ckpt")


def getScope(filename):
    df = pd.read_csv(filename).iloc[:, :-1]
    data60 = df.loc[:60].values
    data200 = df.loc[60:160].values
    R1 = np.mean(model.transform(data60), axis=0)
    R2s = model.transform(data200)
    means = (R2s - R1) ** 2
    mse = np.sum(means, axis=1)
    return min(mse), max(mse)


def getMses(filename):
    df = pd.read_csv(filename).iloc[:, :-1]
    data60 = df.loc[160:220].values
    data200 = df.loc[220:300].values
    R1 = np.mean(model.transform(data60), axis=0)
    R2s = model.transform(data200)
    means = (R2s - R1) ** 2
    mse = np.sum(means, axis=1)
    return mse


path = dataRoot + '\\Data\\csv\\'
print(path)

ball1Scope = getScope(path + 'Ball Fault Level1')
ball2Scope = getScope(path + 'Ball Fault Level2')
ball3Scope = getScope(path + 'Ball Fault Level3')
ball4Scope = getScope(path + 'Ball Fault Level4')
combScope = getScope(path + 'Combination Fault')
innerScope = getScope(path + 'Inner Race Fault')
normScope = getScope(path + 'Normal-Data')
outerScope = getScope(path + 'Outer Race Fault')

print('ball1Scope:', ball1Scope)
print('ball2Scope:', ball2Scope)
print('ball3Scope:', ball3Scope)
print('ball4Scope:', ball4Scope)
print('combScope:', combScope)
print('innerScope:', innerScope)
print('normScope:', normScope)
print('outerScope:', outerScope)

# print('outerScope', outerScope)


Mses = {
    'ball1Mse': getMses(path + 'Ball Fault Level1'),
    'ball2Mse': getMses(path + 'Ball Fault Level2'),
    'ball3Mse': getMses(path + 'Ball Fault Level3'),
    'ball4Mse': getMses(path + 'Ball Fault Level4'),
    'combMse ': getMses(path + 'Combination Fault'),
    'innerMse': getMses(path + 'Inner Race Fault'),
    'normMse': getMses(path + 'Normal-Data'),
    'outerMse': getMses(path + 'Outer Race Fault')
}

correct_num = 0
all_num = 0
for mse in Mses.values():
    for j in mse:
        if j > normScope[0] and j < normScope[1]:
            all_num += 1
        else:
            correct_num += 1
            all_num += 1

print('测试正确的个数:', correct_num)
print('总共测试的记录数:', all_num)
print('准确率为:', correct_num / all_num * 100, '%')
