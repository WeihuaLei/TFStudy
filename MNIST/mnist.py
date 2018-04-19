import input_data
import sys
import tensorflow as tf
_WORKPATH=sys.path

#加载数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.__dict__)


#图像变量
x = tf.placeholder("float", [None, 784])

#权重和bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

#真实label
y_ = tf.placeholder("float", [None, 10])

cross_entropy = tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
#init = tf.initialize_all_tables()
#init = tf.tables_initializer()
init = tf.global_variables_initializer()

#tf session
with tf.Session() as sess:
    sess.run(init)

    #训练模型
    for i in range(10):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print("current is {0} step..".format(i))
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
        #print(sess.run(cross_entropy))

    #预测
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
    print("accuracy of test data is: {0}".format(acc))

