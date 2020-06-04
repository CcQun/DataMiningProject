import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import DataSet


data,label = DataSet.dataset()

num_classes = 8

label = np.eye(num_classes)[label.reshape(-1).astype(np.int8)]

X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0, test_size=0.2)

num_features = 408
num_h1 = 100
num_h2 = 100
learning_rate = 0.001

X = tf.placeholder(tf.float32, shape=(None, num_features))
y = tf.placeholder(tf.float32, shape=(None, num_classes))

weights = {
    'weight_h1': tf.Variable(tf.truncated_normal((num_features, num_h1), stddev=0.1)),
    'weight_h2': tf.Variable(tf.truncated_normal((num_h1, num_h2), stddev=0.1)),
    'weight_ouput': tf.Variable(tf.truncated_normal((num_h2, num_classes), stddev=0.1))
}

biases = {
    'bias_h1': tf.Variable(tf.constant(0.1, shape=[num_h1])),
    'bias_h2': tf.Variable(tf.constant(0.1, shape=[num_h2])),
    'bias_output': tf.Variable(tf.constant(0.1, shape=[num_classes]))
}

h1 = tf.sigmoid(tf.add(tf.matmul(X, weights['weight_h1']), biases['bias_h1']))
h2 = tf.sigmoid(tf.add(tf.matmul(h1, weights['weight_h2']), biases['bias_h2']))
logits = tf.nn.softmax(tf.add(tf.matmul(h2, weights['weight_ouput']), biases['bias_output']))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, float))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

epoch_num = 150
batch_size = 128
total_batch = int(X_train.shape[0] / batch_size)
display_step = 10
record_step = 5

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_summary = []
for epoch_num in range(epoch_num):
    for batch_num in range(total_batch):
        batch_start = batch_num * batch_size
        batch_end = (batch_num + 1) * batch_size
        train_data = X_train[batch_start:batch_end, :]
        train_label = y_train[batch_start:batch_end, :]
        cost, opt = sess.run((loss, optimizer), feed_dict={X: train_data, y: train_label})

    if epoch_num % display_step == 0 or epoch_num % record_step == 0:
        total_cost,total_accuracy = sess.run((loss,accuracy), feed_dict={X: X_train, y: y_train})
        if epoch_num % display_step == 0:
            total_test_cost, total_test_accuracy = sess.run((loss, accuracy), feed_dict={X: X_test, y: y_test})
            print('Epoch{}:'.format(epoch_num + 1))
            print('    Train:cost={:.9f},accuracy={:.5f}'.format(total_cost, total_accuracy))
            print('    Test:cost={:.9f},accuracy={:.5f}'.format(total_test_cost, total_test_accuracy))
        if epoch_num % record_step == 0:
            cost_summary.append({'epoch': epoch_num + 1, 'cost': total_cost})

dataRoot = "D:\\CodingProject\\PyCharmProject\\DataMiningProject\\"
saver = tf.train.Saver()
save_path = dataRoot + '\\model\\MLP\\'
saver.save(sess,save_path + "mlp.ckpt")

f, ax1 = plt.subplots(1, 1, figsize=(10, 4))
ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(map(lambda x: x['cost'], cost_summary)))
ax1.set_title('cost')

plt.xlabel('epoch num')
plt.show()