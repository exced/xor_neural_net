from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

# Hyper parameters
learning_rate = 0.1
training_epochs = 10000
display_step = 100

def main(_):
    # Feed Datas
    x_d = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_d = [[0], [1], [1], [0]]
    # Create the model
    x_ = tf.placeholder(tf.float32, [4, 2])  # correct input
    y_ = tf.placeholder(tf.float32, [4, 1])  # correct output
    b1 = tf.Variable(tf.zeros([2]))  # bias
    b2 = tf.Variable(tf.zeros([1]))  # bias
    t1 = tf.Variable(tf.random_uniform([2, 2], -1, 1))  # theta1
    t2 = tf.Variable(tf.random_uniform([2, 1], -1, 1))  # theta2
    layer2 = tf.sigmoid(tf.matmul(x_, t1) + b1)
    y = tf.sigmoid(tf.matmul(layer2, t2) + b2)  # training output

    cost = tf.reduce_mean(-((y_ * tf.log(y)) + ((1 - y_) * tf.log(1.0 - y)))) # cost entropy
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # optimizer

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for i in range(training_epochs):
        sess.run(train_step, feed_dict={x_: x_d, y_: y_d})
        if i % display_step == 0:
            print('cost ', sess.run(cost, feed_dict={x_: x_d, y_: y_d}))
    print('y ', sess.run(y, feed_dict={x_: x_d, y_: y_d}))

if __name__ == '__main__':
    tf.app.run(main=main)
