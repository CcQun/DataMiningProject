import os
import numpy as np
import tensorflow as tf
# 为显示中文，导入中文字符集
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf')
import matplotlib.pyplot as plt
import pandas as pd

dataRoot = "D:\\CodingProject\\PyCharmProject\\DataMiningProject\\Data\\csv"


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
            tf.add(tf.matmul(X + self.training_scale * tf.random_normal((self.n_input,)), self.weights['encoder_h1']),
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


# 获取训练数据
X_train = pd.read_csv(os.path.join(dataRoot, 'Normal-Data')).iloc[:, :-1].values
print(X_train.shape)

# 创建模型
model = Autoencoder(n_hidden_1=210, n_hidden_2=60, n_input=X_train.shape[1], learning_rate=0.01)

# 定义训练步数、批量大小等超参数
training_epochs = 324
batch_size = 60
display_step = 5
record_step = 5

# 训练模型
total_batch = int(X_train.shape[0] / batch_size)

cost_summary = []

print("开始训练")
for epoch in range(training_epochs):
    cost = None
    for i in range(total_batch):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = X_train[batch_start:batch_end, :]

        cost = model.partial_fit(batch)

    if epoch % display_step == 0 or epoch % record_step == 0:
        total_cost = model.calc_total_cost(X_train)

        if epoch % record_step == 0:
            cost_summary.append({'epoch': epoch + 1, 'cost': total_cost})

        if epoch % display_step == 0:
            print("Epoch:{}, cost={:.9f}".format(epoch + 1, total_cost))

'''
7．查看迭代步数与损失值的关系
'''
f, ax1 = plt.subplots(1, 1, figsize=(10, 4))
ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(map(lambda x: x['cost'], cost_summary)))
ax1.set_title('损失值', fontproperties=myfont)

plt.xlabel('迭代次数', fontproperties=myfont)
plt.show()


# def test(testDataPath,cycleNum):
#     #测试模型
#     encode_decode = None
#     X_test = getTestData(testDataPath,cycleNum)
#     total_batch = int(X_test.shape[0]/batch_size) + 1
#     for i in range(total_batch):
#         batch_start = i * batch_size
#         batch_end = (i + 1) * batch_size
#         batch = X_test[batch_start:batch_end, :]
#         batch_res = model.reconstruct(batch)
#         if encode_decode is None:
#             encode_decode = batch_res
#         else:
#             encode_decode = np.vstack((encode_decode, batch_res)) #np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
#     df = get_df(X_test, encode_decode)
#
#     # 查看指标的统计信息
#     print(df)



# # 获取性能指标
# def get_df(orig, ed):
#     # rmse = np.mean(np.power(orig - ed, 2), axis=1)
#     rmse = np.mean(np.power(orig - ed, 2))
#     # return pd.DataFrame({'rmse': rmse})
#     return rmse
#
#
# def getR1():
#     encode_decode = None
#     X_test = getTestData(normal + "Normal-0-1-12-1.txt", 60)
#     total_batch = int(X_test.shape[0] / batch_size) + 1
#     for i in range(total_batch):
#         batch_start = i * batch_size
#         batch_end = (i + 1) * batch_size
#         batch = X_test[batch_start:batch_end, :]
#         batch_res = model.reconstruct(batch)
#         if encode_decode is None:
#             encode_decode = batch_res
#         else:
#             encode_decode = np.vstack((encode_decode, batch_res))  # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
#     rmse = np.mean(np.power(X_test - encode_decode, 2))
#     return rmse
#
#
# def getR2(testDataPath):
#     encode_decode = None
#     X_test = getTestData(testDataPath, 200)
#     total_batch = int(X_test.shape[0] / batch_size) + 1
#     for i in range(total_batch):
#         batch_start = i * batch_size
#         batch_end = (i + 1) * batch_size
#         batch = X_test[batch_start:batch_end, :]
#         batch_res = model.reconstruct(batch)
#         if encode_decode is None:
#             encode_decode = batch_res
#         else:
#             encode_decode = np.vstack((encode_decode, batch_res))  # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
#     rmse = np.mean(np.power(X_test - encode_decode, 2), axis=1)
#     return rmse
#
#
# def getMinAndMax(R1, testDataPath):
#     AllR2 = getR2(testDataPath)
#     # result = []
#     # for i in range(len(AllR2)):
#     #     result.append(np.power(AllR2[i] - R1, 2) / 2)
#     # return min(result),max(result)
#     mean = np.mean(np.power(AllR2 - R1, 2) / 2)
#     return mean
#
#
# R1 = getR1()
# print("正常状态测试" + str(getMinAndMax(R1, normal + "Normal-0-1-12-1.txt")))
# print("正常状态测试" + str(getMinAndMax(R1, normal + "Normal-0-1-12-2.txt")))
# print()
# print("滚珠轻微故障状态测试" + str(getMinAndMax(R1, ballFaultLevel1 + "Comb-Ball1-5-1-12-A1B1-Same-2.txt")))
# print("滚珠轻微故障状态测试" + str(getMinAndMax(R1, ballFaultLevel1 + "Comb-Ball1-5-1-12-A1B1-Same-3.txt")))
# print()
# print("滚珠轻度故障状态测试" + str(getMinAndMax(R1, ballFaultLevel2 + "Comb-Ball2-5-6-12-A1B1-Same-1.txt")))
# print("滚珠轻度故障状态测试" + str(getMinAndMax(R1, ballFaultLevel2 + "Comb-Ball2-5-6-12-A1B1-Same-2.txt")))
# print()
# print("滚珠中度故障状态测试" + str(getMinAndMax(R1, ballFaultLevel3 + "Comb-Ball3-5-11-12-A1B1-Same-2.txt")))
# print("滚珠中度故障状态测试" + str(getMinAndMax(R1, ballFaultLevel3 + "Comb-Ball3-5-11-12-A1B1-Same-3.txt")))
# print()
# print("滚珠重度故障状态测试" + str(getMinAndMax(R1, ballFaultLevel4 + "Comb-Ball4-5-16-12-A1B1-Same-1.txt")))
# print("滚珠重度故障状态测试" + str(getMinAndMax(R1, ballFaultLevel4 + "Comb-Ball4-5-16-12-A1B1-Same-2.txt")))
# print()
# print("内圈故障状态测试" + str(getMinAndMax(R1, innerRaceFault + "Comb-Inner-5-21-12-A1B1-Same-1.txt")))
# print("内圈故障状态测试" + str(getMinAndMax(R1, innerRaceFault + "Comb-Inner-5-21-12-A1B1-Same-2.txt")))
# print()
# print("外圈故障状态测试" + str(getMinAndMax(R1, outerRaceFault + "Comb-Outer-5-26-12-A1B1-Same-1.txt")))
# print("外圈故障状态测试" + str(getMinAndMax(R1, outerRaceFault + "Comb-Outer-5-26-12-A1B1-Same-2.txt")))
# print()
# print("混合故障状态测试" + str(getMinAndMax(R1, combinationFault + "Comb-Comb-5-31-12-A1B1-Same-1.txt")))
# print("混合故障状态测试" + str(getMinAndMax(R1, combinationFault + "Comb-Comb-5-31-12-A1B1-Same-2.txt")))

# print("正常状态测试")
# test(normal + "Normal-0-1-12-1.txt",300)
#
# print("滚珠轻微故障状态测试")
# test(ballFaultLevel1 + "Comb-Ball1-5-1-12-A1B1-Same-2.txt",300)
#
# print("滚珠轻度故障状态测试")
# test(ballFaultLevel2 + "Comb-Ball2-5-6-12-A1B1-Same-1.txt",300)
#
# print("滚珠中度故障状态测试")
# test(ballFaultLevel3 + "Comb-Ball3-5-11-12-A1B1-Same-1.txt",300)
#
# print("滚珠重度故障状态测试")
# test(ballFaultLevel4 + "Comb-Ball4-5-16-12-A1B1-Same-1.txt",300)
#
# print("内圈故障状态测试")
# test(innerRaceFault + "Comb-Inner-5-21-12-A1B1-Same-1.txt",300)
#
# print("外圈故障状态测试")
# test(outerRaceFault + "Comb-Outer-5-26-12-A1B1-Same-1.txt",300)
#
# print("混合故障状态测试")
# test(combinationFault + "Comb-Comb-5-31-12-A1B1-Same-1.txt",300)
#
#
#
