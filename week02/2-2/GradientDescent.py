import tensorflow as tf
import numpy as np

#简单的梯度下降
def test1():
    #使用numpy 生成100个随机点
    x_data = np.random.rand(100)
    y_data = x_data*0.1 + 2

    b = tf.Variable(0.)
    k = tf.Variable(0.)
    y = k*x_data + b

    #二次代价函数
    loss = tf.reduce_mean(tf.square(y_data-y))
    #定义一个梯度下降法来进行训练的优化器
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    #最小化代价函数
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(1001):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))

if __name__ == '__main__':
    test1()