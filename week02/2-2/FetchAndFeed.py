import tensorflow as tf

#简单的梯度下降
#Fetch 可以传入一些tensor来传回运行结果
def fetch():
    input = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)

    add = tf.add(input2,input3)
    mul = tf.multiply(input,add)

    sess = tf.Session()
    result = sess.run([mul,add])
    print(result)

#Feed feed机制可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.
def feed():
    # 创建占位符
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1,input2)
    sess = tf.Session()
    result = sess.run(output,feed_dict={input1:[8.0],input2:[2.0]})
    print(result)

if __name__ == '__main__':
    fetch()
    feed()

