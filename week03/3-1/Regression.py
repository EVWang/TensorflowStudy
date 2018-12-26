import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# def test():
# numpy  [-0.5,0.5] 生成200个点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
# y = x^2 + noise
y_data = np.square(x_data)+noise


#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32, [None, 1])

#构建神经网络
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biase_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biase_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经元输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
# 偏置值
biase_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biase_L2
# 预测值
prediction  = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 定义一个梯度下降法来进行训练的优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()



