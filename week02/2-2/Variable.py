import tensorflow as tf

def test():
    x = tf.Variable([1,2])
    a = tf.Variable([3,2])
    # 增加一个减法的op
    sub = tf.subtract(x,a)
    # 增加一个加法的op
    add = tf.add(x,a)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

def test1():
    state = tf.Variable(0,name='counter')
    #创建一个op 把state 加1
    new_value = tf.add(state,1)
    update = tf.assign(state,new_value)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))

if __name__ == '__main__':
    # test()
    test1()
