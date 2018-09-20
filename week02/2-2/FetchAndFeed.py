import tensorflow as tf

#Fetch
def fetch():
    input = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)

    add = tf.add(input2,input3)
    mul = tf.multiply(input,add)

    sess = tf.Session()
    result = sess.run([mul,add])
    print(result)

#Feed
def feed():
    # 创建占位符
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1,input2)

    sess = tf.Session()
    print(sess.run(output,feed_dict={input1:[8.0],input2:[2.0]}))

if __name__ == '__main__':
    # fetch()
    feed()
