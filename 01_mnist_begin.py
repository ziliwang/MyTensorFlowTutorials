# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# __future__ 引用新特性，代码可能是python2移植过来的

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])  # 占位符，？x784矩阵，784为像素
W = tf.Variable(tf.zeros([784, 10]))  # 权重矩阵，784x10矩阵
b = tf.Variable(tf.zeros([10]))  # 偏移向量，1x10
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 模型，单层神经网络

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])  # 占位符，实际y值，？x10
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])  # 损失函数
)  # 减小损失函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#  梯度下降，学习率为0.5，最小化损失函数
# Train
tf.initialize_all_variables().run()  # 所以参数初始化
for i in range(1000):  # 1000步
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每步从训练集随机提取100个样本
    train_step.run({x: batch_xs, y_: batch_ys})  # 训练

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#  判断预测与现实是否相同，返回[1,0,1,1]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
