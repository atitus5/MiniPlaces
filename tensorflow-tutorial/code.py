
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
data_dir = 'MNIST_data/'
mnist = read_data_sets(data_dir, one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
data_dir = 'MNIST_data/'
mnist = read_data_sets(data_dir, one_hot=True)

# construct placeholders as the input ports to the graph
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#print x.__dict__

# define network trainable parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#print W.__dict__

# define graph operations
y_ = tf.nn.softmax(tf.matmul(x, W) + b)


# define loss (cross entropy loss -\sum y * log(y_))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))


# define optimizer for training
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)


# define the operation that initializes variables
init = tf.global_variables_initializer()



# Launch the graph
with tf.Session() as sess:
    # initialization
    sess.run(init)
    
    # Train for 1000 iterations
    batch_size = 100
    training_iters = 1000
    for i in range(training_iters):
        # load a batch of data
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # feed data into placeholder, run optimizer 
        _ = sess.run([train_optimizer], feed_dict={x: batch_x, y: batch_y})
        
        if i%100 == 0:
            print('Training iterations:', i)
    
    # Evaluate the trained model
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Calculate accuracy for 500 mnist test images
    accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images[:500], y: mnist.test.labels[:500]})
    print('Testing Accuracy:', accuracy_val)



