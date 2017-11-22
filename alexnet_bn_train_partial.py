import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader_partial import *
import operator

# Dataset Parameters
batch_size = 100
load_size = 128
fine_size = 128
c = 3

# Note: may want to actually recompute this, or just do away with it...
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 1000
step_save = 1000
path_save = './saved_models/random_size_1000'
start_from = ''

test_only = False

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
def alexnet(x, keep_dropout, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([8*8*64, 4096], stddev=np.sqrt(2./(8*8*64)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # Conv + ReLU + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU  + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.matmul(fc6, weights['wf6'])
    fc6 = batch_norm_layer(fc6, train_phase, 'bn6')
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.matmul(fc6, weights['wf7'])
    fc7 = batch_norm_layer(fc7, train_phase, 'bn7')
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_128_train.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True,
    'count': 10000,

    }
opt_data_val = {
    'data_h5': 'miniplaces_128_val.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }
opt_data_test = {
    'data_h5': 'miniplaces_128_test.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

# loader_train = DataLoaderDisk(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)

print("Loading validation data...")
loader_val = DataLoaderH5(**opt_data_val)

print("Loading test data...")
loader_test = DataLoaderH5(**opt_data_test)

val_guesses = np.zeros(shape=(10000,100))

val_correct = []
with open('development_kit/data/val.txt') as f:
    for line in f:
        val_correct.append(int(line.split()[1]))

def evaluate_guesses(label_guesses):
    acc_1_sum = 0
    acc_5_sum = 0
    pos_sum = 0
    for i in range(10000):
        correct = val_correct[i]
        lis = list(label_guesses[i])
        pos_sum += lis.index(correct)
        if correct == lis[0]:
            acc_1_sum += 1
            acc_5_sum += 1
        elif correct in lis[0:5]:
            acc_5_sum += 1
    print('Accuracy Top1 = ' + "{:.4f}".format(acc_1_sum/10000.0) + ", Top5 = " + "{:.4f}".format(acc_5_sum/10000.0) + " Average place = " + "{:.4f}".format(pos_sum/10000.0))
    
# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = alexnet(x, keep_dropout, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

predict = tf.nn.top_k(logits, 5)
predict_all = tf.nn.top_k(logits, 100)

for i in range(19):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, path_save+"_"+str(i)+"-"+str(training_iters))
        num_batch = loader_val.size()//batch_size
        acc1_total = 0.
        acc5_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            images_batch, labels_batch = loader_val.next_batch(batch_size)    
            acc1, acc5, probs = sess.run([accuracy1, accuracy5, logits], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
            acc1_total += acc1
            acc5_total += acc5
            val_guesses[i*batch_size:(i+1)*batch_size,:] += probs

        acc1_total /= num_batch
        acc5_total /= num_batch
        print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
        label_guesses = np.argsort(-val_guesses)
        evaluate_guesses(label_guesses)  

if not test_only:
    for i in range(19,1000):
        
        print("Loading train data...")
        loader_train = DataLoaderH5(**opt_data_train)
        # define initialization
        init = tf.global_variables_initializer()

        # define saver
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            step = 0

            if not test_only:
                while step < training_iters:
                    # Load a batch of training data
                    images_batch, labels_batch = loader_train.next_batch(batch_size)
                    
                    # Run optimization op (backprop)
                    sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, train_phase: True})
                    
                    step += 1
                    
                    # Save model
                    if step % step_save == 0:
                        saver.save(sess, path_save+"_"+str(i), global_step=step)           

                
                # Evaluate on the whole validation set
            print('Evaluation on the whole validation set...')
            
            num_batch = loader_val.size()//batch_size
            acc1_total = 0.
            acc5_total = 0.
            loader_val.reset()
            for i in range(num_batch):
                images_batch, labels_batch = loader_val.next_batch(batch_size)    
                acc1, acc5, probs = sess.run([accuracy1, accuracy5, logits], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
                acc1_total += acc1
                acc5_total += acc5
                val_guesses[i*batch_size:(i+1)*batch_size,:] += probs

            acc1_total /= num_batch
            acc5_total /= num_batch
            print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
            label_guesses = np.argsort(-val_guesses)
            evaluate_guesses(label_guesses)      
            
            

            
        """
                # Evaluate on the whole validation set
            print('results on the test set')
            num_batch = loader_test.size()//batch_size
            loader_test.reset()
            with open("pred.txt", "w") as f:
                for i in range(num_batch):
                    images_batch, labels_batch = loader_test.next_batch(batch_size)
                    labels = sess.run(predict, feed_dict={x: images_batch, keep_dropout: 1., train_phase: False})[1]
                    for item in labels:
                        f.write("test/"+str(i+1).zfill(8)+".jpg "+ " ".join([str(it) for it in item])+"\n")
        """