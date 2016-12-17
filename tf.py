from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def main(_):

  # Create the model
  x_ = tf.placeholder(tf.float32, [4, 2]) #correct input
  y_ = tf.placeholder(tf.float32, [4, 1]) #correct output
  b1 = tf.Variable(tf.zeros([2])) #bias
  b2 = tf.Variable(tf.zeros([1])) #bias 
  t1 = tf.Variable(tf.random_uniform([2,2], -1, 1)) #theta1
  t2 = tf.Variable(tf.random_uniform([2,1], -1, 1)) #theta2
  layer2 = tf.sigmoid(tf.matmul(x_, t1) + b1)
  y = tf.sigmoid(tf.matmul(layer2, t2) + b2) #training output

  loss = tf.nn.l2_loss(y_ - y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(10000):
    sess.run(train_step, feed_dict={x_: [[0,0],[0,1],[1,0],[1,1]], y_: [[0],[1],[1],[0]]})
    if i % 100 == 0:
		  print('y ', sess.run(y, feed_dict={x_: [[0,0],[0,1],[1,0],[1,1]], y_: [[0],[1],[1],[0]]}))
		  print('cost ', sess.run(cost, feed_dict={x_: [[0,0],[0,1],[1,0],[1,1]], y_: [[0],[1],[1],[0]]}))

if __name__ == '__main__':
  tf.app.run(main=main)