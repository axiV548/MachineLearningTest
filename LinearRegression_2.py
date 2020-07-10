# -*- coding: utf-8 -*-
# author：albert time:2020/5/18
#Tensorflow2.0 线性回归方程，小批量随机线性回归
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

print("TF.version:", tf.__version__)
x_data = np.linspace(-1, 1, 100)
print(x_data)
np.random.seed(5)
y_data = 2 * x_data + 1.0 + np.random.randn(*x_data.shape) * 0.4
print(y_data)

plt.scatter(x_data, y_data)
plt.plot(x_data, 1.0 + 2 * x_data, 'r', linewidth=3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("FI")

b = tf.Variable(0.0, tf.float32)
w = tf.Variable(np.random.randn(), tf.float32)

def model(x, w, b):
    return tf.multiply(x, w) + b

def loss(x, y, w, b):
    err = model(x, w, b) - y
    squared_err = tf.square(err)
    return tf.reduce_mean(squared_err)

training_epochs = 100
learning_rate = 0.05

def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])

loss_list = []

for epoch in range(training_epochs):
    loss_ = loss(x_data, y_data, w ,b)
    loss_list.append(loss_)

    delta_w, delta_b = grad(x_data, y_data, w, b)
    change_w = delta_w * learning_rate
    change_b = delta_b * learning_rate
    w.assign_sub(change_w)
    b.assign_sub(change_b)

    print("TE", '%02d' % (epoch + 1), 'loss=%.6f' % (loss_))
    plt.plot(x_data, w.numpy() * x_data + b.numpy())
plt.show()
plt.close()
plt.plot(loss_list)
plt.show()