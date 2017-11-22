import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader_multi import *




vgg = True
alex = True

write = True

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

def write_guesses(label_guesses):
    with open("pred5.txt","w") as f:
        for i in range(10000):
            f.write("test/"+str(i + 1).zfill(8)+".jpg "+ " ".join([str(it) for it in list(label_guesses[i])])[0:5]+"\n")
        


if vgg: # vgg
    # Dataset Parameters
    load_size = 128
    fine_size = 100
    c = 3

    # Note: may want to actually recompute this, or just do away with it...
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
    # Construct dataloader
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

    print("Loading validation data...", flush=True)
    loader_val = DataLoaderH5(**opt_data_val)

    print("Loading test data...", flush=True)
    loader_test = DataLoaderH5(**opt_data_test)
    path_save = './saved_models/vgg_A_bn_multi'
    start_from = '157000'


    def batch_norm_layer(x, train_phase, scope_bn):
        return batch_norm(x, decay=0.9, center=True, scale=True,
        updates_collections=None,
        is_training=train_phase,
        reuse=None,
        trainable=True,
        scope=scope_bn)
        
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=np.sqrt(2./(3*3*3)))),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2./(3*3*64)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*128)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(3*3*256)))),
        'wc6': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc7': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc8': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),

        'wf9': tf.Variable(tf.random_normal([8192, 4096], stddev=np.sqrt(2./8192))),
        'wf10': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros(64)),
        'bc2': tf.Variable(tf.zeros(128)),
        'bc3': tf.Variable(tf.zeros(256)),
        'bc4': tf.Variable(tf.zeros(256)),
        'bc5': tf.Variable(tf.zeros(512)),
        'bc6': tf.Variable(tf.zeros(512)),
        'bc7': tf.Variable(tf.zeros(512)),
        'bc8': tf.Variable(tf.zeros(512)),

        'bf9': tf.Variable(tf.zeros(4096)),
        'bf10': tf.Variable(tf.zeros(4096)),
        'bo': tf.Variable(tf.zeros(100))
    }

    def vgg_A(x, keep_dropout, train_phase):
        # Conv3-64 + ReLU + 2x2 Pool
        conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Conv3-128 + ReLU + 2x2 Pool
        conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Conv3-256 + ReLU + Conv3-256 + ReLU + 2x2 Pool
        conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
        conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))
        conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
        conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
        conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))
        pool3 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Conv3-512 + ReLU + Conv3-512 + ReLU + 2x2 Pool
        conv5 = tf.nn.conv2d(pool3, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
        conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
        conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
        conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
        conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
        conv6 = tf.nn.relu(tf.nn.bias_add(conv6, biases['bc6']))
        pool4 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Conv3-512 + ReLU + Conv3-512 + ReLU + 2x2 Pool
        conv7 = tf.nn.conv2d(pool4, weights['wc7'], strides=[1, 1, 1, 1], padding='SAME')
        conv7 = batch_norm_layer(conv7, train_phase, 'bn7')
        conv7 = tf.nn.relu(tf.nn.bias_add(conv7, biases['bc7']))
        conv8 = tf.nn.conv2d(conv7, weights['wc8'], strides=[1, 1, 1, 1], padding='SAME')
        conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
        conv8 = tf.nn.relu(tf.nn.bias_add(conv8, biases['bc8']))
        pool5 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # FC + ReLU + Dropout
        fc9 = tf.reshape(pool5, [-1, weights['wf9'].get_shape().as_list()[0]])
        fc9 = tf.add(tf.matmul(fc9, weights['wf9']), biases['bf9'])
        fc9 = batch_norm_layer(fc9, train_phase, 'bn9')
        fc9 = tf.nn.relu(fc9)
        fc9 = tf.nn.dropout(fc9, keep_dropout)
        
        # FC + ReLU + Dropout
        fc10 = tf.add(tf.matmul(fc9, weights['wf10']), biases['bf10'])
        fc10 = batch_norm_layer(fc10, train_phase, 'bn10')
        fc10 = tf.nn.relu(fc10)
        fc10 = tf.nn.dropout(fc10, keep_dropout)

        # Output FC
        out = tf.add(tf.matmul(fc10, weights['wo']), biases['bo'])
        
        return out



    # tf Graph input
    x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
    y = tf.placeholder(tf.int64, None)
    keep_dropout = tf.placeholder(tf.float32)
    train_phase = tf.placeholder(tf.bool)

    # Construct model
    logits = vgg_A(x, keep_dropout, train_phase)

    # Define loss and optimizer
    l2_regularizer= sum(tf.nn.l2_loss(weights[key]) for key in weights.keys())
    lr_placeholder = tf.placeholder(tf.float32, [])
    
    # Evaluate model
    accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
    accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

    predict = tf.nn.top_k(logits, 5)

    # define initialization
    init = tf.global_variables_initializer()

    # define saver (only saves best so far)
    saver = tf.train.Saver(max_to_keep=1)

    # define summary writer
    #writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

    # Launch the graph
    with tf.Session() as sess:
        # Initialization
        current_lr = .00001
        if len(start_from)>1:
            saver.restore(sess, path_save+"-"+start_from)
            step = int(start_from)
        else:
            sess.run(init)
            step = 0

        if not write:
            # Evaluate on the whole validation set
            print('Evaluation on the whole validation set...', flush=True)
            batch_size = 5
            num_batch = loader_val.size()//batch_size
            acc1_total = 0.
            acc5_total = 0.
            loader_val.reset()
            for i in range(num_batch):
                if i %25 == 0:
                    print(i, "out of", num_batch)
                images_batch, labels_batch = loader_val.next_batch_all(batch_size)    
                acc1, acc5, probs = sess.run([accuracy1, accuracy5, logits], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., lr_placeholder: current_lr, train_phase: False})
                acc1_total += acc1
                acc5_total += acc5
                for j in range(batch_size):
                    val_guesses[i*batch_size+j,:] += np.average(probs[9*j:9*(j+1)], axis=0)
                if i %25 == 0:
                    print("Validation Accuracy Top1 = " + \
                      "{:.4f}".format(acc1_total/(i+1)) + ", Top5 = " + \
                      "{:.4f}".format(acc5_total/(i+1)), flush=True)

            acc1_total /= num_batch
            acc5_total /= num_batch
            print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total), flush=True)
            label_guesses = np.argsort(-val_guesses)
            evaluate_guesses(label_guesses)    
        else:
            print('Evaluation on the whole test set...')
            batch_size = 5
            num_batch = loader_test.size()//batch_size
            loader_test.reset()
            for i in range(num_batch):
                if i %25 == 0:
                    print(i, "out of", num_batch)
                images_batch, labels_batch = loader_test.next_batch_all(batch_size)    
                probs = sess.run(logits, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
                for j in range(batch_size):
                    val_guesses[i*batch_size+j,:] += np.average(probs[9*j:9*(j+1)], axis=0)
tf.reset_default_graph() 
if alex:
    # Dataset Parameters
    load_size = 128
    fine_size = 96
    c = 3

    # Note: may want to actually recompute this, or just do away with it...
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
    # Construct dataloader
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

    print("Loading validation data...", flush=True)
    loader_val = DataLoaderH5(**opt_data_val)


    print("Loading test data...", flush=True)
    loader_test = DataLoaderH5(**opt_data_test)

    # Training Parameters
    learning_rate = 0.0001
    dropout = 0.5 # Dropout, probability to keep units
    path_save = './saved_models/alexnet_bn_multi'
    start_from = '165000'

    test_only = True

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

            'wf6': tf.Variable(tf.random_normal([9*256, 4096], stddev=np.sqrt(2./(8*8*64)))),
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


    # define initialization
    init = tf.global_variables_initializer()

    # define saver
    saver = tf.train.Saver()

    # define summary writer
    #writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

    # Launch the graph
    with tf.Session() as sess:
        # Initialization
        if len(start_from)>1:
            saver.restore(sess, path_save+"-"+start_from)
            step = int(start_from)
        else:
            sess.run(init)
            step = 0

        
            
        # Evaluate on the whole validation set
        if not write:
            print('Evaluation on the whole validation set...')
            batch_size = 20
            num_batch = loader_val.size()//batch_size
            acc1_total = 0.
            acc5_total = 0.
            loader_val.reset()
            for i in range(num_batch):
                if i %25 == 0:
                    print(i, "out of", num_batch)
                images_batch, labels_batch = loader_val.next_batch_all(batch_size)    
                acc1, acc5, probs = sess.run([accuracy1, accuracy5, logits], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
                acc1_total += acc1
                acc5_total += acc5
                for j in range(batch_size):
                    val_guesses[i*batch_size+j,:] += np.average(probs[9*j:9*(j+1)], axis=0)
                if i %25 == 0:
                    print("Validation Accuracy Top1 = " + \
                      "{:.4f}".format(acc1_total/(i+1)) + ", Top5 = " + \
                      "{:.4f}".format(acc5_total/(i+1)))

            acc1_total /= num_batch
            acc5_total /= num_batch
            print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))
            label_guesses = np.argsort(-val_guesses)
            evaluate_guesses(label_guesses)  
        else:
            print('Evaluation on the whole test set...')
            batch_size = 20
            num_batch = loader_test.size()//batch_size
            loader_test.reset()
            for i in range(num_batch):
                if i %25 == 0:
                    print(i, "out of", num_batch)
                images_batch, labels_batch = loader_test.next_batch_all(batch_size)    
                probs = sess.run(logits, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., train_phase: False})
                for j in range(batch_size):
                    val_guesses[i*batch_size+j,:] += np.average(probs[9*j:9*(j+1)], axis=0)



if write:
    label_guesses = np.argsort(-val_guesses)
    write_guesses(label_guesses)   

