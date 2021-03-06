import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *

# Dataset Parameters
batch_size = 100
load_size = 128
fine_size = 100
c = 3

# Note: may want to actually recompute this, or just do away with it...
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
init_lr = 0.01
lr_decay_factor = 10
max_decays = 3
momentum = 0.9
beta = 0.0005

dropout = 0.5 # Dropout, probability to keep units
num_epochs = 75
training_iters = (100000 / batch_size) * num_epochs     # Based on VGG paper
step_display = 50
step_save = training_iters / num_epochs     # Once per epoch
path_save = './saved_models/vgg_E_bn'
start_from = ''

eval_only = False

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)
    
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=np.sqrt(2./(3*3*3)))),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2./(3*3*64)))),
    'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
    'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*128)))),
    'wc6': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
    'wc7': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
    'wc8': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
    'wc9': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(3*3*256)))),
    'wc10': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
    'wc11': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
    'wc12': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
    'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
    'wc14': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
    'wc15': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
    'wc16': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),

    'wf17': tf.Variable(tf.random_normal([8192, 4096], stddev=np.sqrt(2./8192))),
    'wf18': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
    'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
}

biases = {
    'bc1': tf.Variable(tf.zeros(64)),
    'bc2': tf.Variable(tf.zeros(64)),
    'bc3': tf.Variable(tf.zeros(128)),
    'bc4': tf.Variable(tf.zeros(128)),
    'bc5': tf.Variable(tf.zeros(256)),
    'bc6': tf.Variable(tf.zeros(256)),
    'bc7': tf.Variable(tf.zeros(256)),
    'bc8': tf.Variable(tf.zeros(256)),
    'bc9': tf.Variable(tf.zeros(512)),
    'bc10': tf.Variable(tf.zeros(512)),
    'bc11': tf.Variable(tf.zeros(512)),
    'bc12': tf.Variable(tf.zeros(512)),
    'bc13': tf.Variable(tf.zeros(512)),
    'bc14': tf.Variable(tf.zeros(512)),
    'bc15': tf.Variable(tf.zeros(512)),
    'bc16': tf.Variable(tf.zeros(512)),

    'bf17': tf.Variable(tf.zeros(4096)),
    'bf18': tf.Variable(tf.zeros(4096)),
    'bo': tf.Variable(tf.zeros(100))
}

def vgg_E(x, keep_dropout, train_phase):
    # Conv3-64 + ReLU + Conv3-64 + ReLU + 2x2 Pool
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # Conv3-128 + ReLU + Conv3-128 + ReLU + 2x2 Pool
    conv3 = tf.nn.conv2d(pool1, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))
    pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # 4x(Conv3-256 + ReLU) + 2x2 Pool
    conv5 = tf.nn.conv2d(pool2, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
    conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
    conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
    conv6 = tf.nn.relu(tf.nn.bias_add(conv6, biases['bc6']))
    conv7 = tf.nn.conv2d(conv6, weights['wc7'], strides=[1, 1, 1, 1], padding='SAME')
    conv7 = batch_norm_layer(conv7, train_phase, 'bn7')
    conv7 = tf.nn.relu(tf.nn.bias_add(conv7, biases['bc7']))
    conv8 = tf.nn.conv2d(conv7, weights['wc8'], strides=[1, 1, 1, 1], padding='SAME')
    conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
    conv8 = tf.nn.relu(tf.nn.bias_add(conv8, biases['bc8']))
    pool3 = tf.nn.max_pool(conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # 4x(Conv3-512 + ReLU) + 2x2 Pool
    conv9 = tf.nn.conv2d(pool3, weights['wc9'], strides=[1, 1, 1, 1], padding='SAME')
    conv9 = batch_norm_layer(conv9, train_phase, 'bn9')
    conv9 = tf.nn.relu(tf.nn.bias_add(conv9, biases['bc9']))
    conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=[1, 1, 1, 1], padding='SAME')
    conv10 = batch_norm_layer(conv10, train_phase, 'bn10')
    conv10 = tf.nn.relu(tf.nn.bias_add(conv10, biases['bc10']))
    conv11 = tf.nn.conv2d(conv10, weights['wc11'], strides=[1, 1, 1, 1], padding='SAME')
    conv11 = batch_norm_layer(conv11, train_phase, 'bn11')
    conv11 = tf.nn.relu(tf.nn.bias_add(conv11, biases['bc11']))
    conv12 = tf.nn.conv2d(conv11, weights['wc12'], strides=[1, 1, 1, 1], padding='SAME')
    conv12 = batch_norm_layer(conv12, train_phase, 'bn12')
    conv12 = tf.nn.relu(tf.nn.bias_add(conv12, biases['bc12']))
    pool4 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 4x(Conv3-512 + ReLU) + 2x2 Pool
    conv13 = tf.nn.conv2d(pool4, weights['wc13'], strides=[1, 1, 1, 1], padding='SAME')
    conv13 = batch_norm_layer(conv13, train_phase, 'bn13')
    conv13 = tf.nn.relu(tf.nn.bias_add(conv13, biases['bc13']))
    conv14 = tf.nn.conv2d(conv13, weights['wc14'], strides=[1, 1, 1, 1], padding='SAME')
    conv14 = batch_norm_layer(conv14, train_phase, 'bn14')
    conv14 = tf.nn.relu(tf.nn.bias_add(conv14, biases['bc14']))
    conv15 = tf.nn.conv2d(conv14, weights['wc15'], strides=[1, 1, 1, 1], padding='SAME')
    conv15 = batch_norm_layer(conv15, train_phase, 'bn15')
    conv15 = tf.nn.relu(tf.nn.bias_add(conv15, biases['bc15']))
    conv16 = tf.nn.conv2d(conv15, weights['wc16'], strides=[1, 1, 1, 1], padding='SAME')
    conv16 = batch_norm_layer(conv16, train_phase, 'bn16')
    conv16 = tf.nn.relu(tf.nn.bias_add(conv16, biases['bc16']))
    pool5 = tf.nn.max_pool(conv16, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc17 = tf.reshape(pool5, [-1, weights['wf17'].get_shape().as_list()[0]])
    fc17 = tf.add(tf.matmul(fc17, weights['wf17']), biases['bf17'])
    fc17 = batch_norm_layer(fc17, train_phase, 'bn17')
    fc17 = tf.nn.relu(fc17)
    fc17 = tf.nn.dropout(fc17, keep_dropout)
    
    # FC + ReLU + Dropout
    fc18 = tf.add(tf.matmul(fc17, weights['wf18']), biases['bf18'])
    fc18 = batch_norm_layer(fc18, train_phase, 'bn18')
    fc18 = tf.nn.relu(fc18)
    fc18 = tf.nn.dropout(fc18, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc18, weights['wo']), biases['bo'])
    
    return out

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_128_train.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
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
if not eval_only:
    print("Loading train data...", flush=True)
    loader_train = DataLoaderH5(**opt_data_train)

print("Loading validation data...", flush=True)
loader_val = DataLoaderH5(**opt_data_val)

print("Loading test data...", flush=True)
loader_test = DataLoaderH5(**opt_data_test)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = vgg_E(x, keep_dropout, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
l2_regularizer= sum(tf.nn.l2_loss(weights[key]) for key in weights.keys())
loss = tf.reduce_mean(loss + beta * l2_regularizer)
lr_placeholder = tf.placeholder(tf.float32, [])
train_optimizer = tf.train.MomentumOptimizer(learning_rate=lr_placeholder, momentum=momentum).minimize(loss)

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
    current_lr = init_lr
    if len(start_from)>1:
        saver.restore(sess, path_save+"-"+start_from)
        step = int(start_from)
    else:
        sess.run(init)
        step = 0

    if not eval_only:
        best_val_loss = float('inf')
        val_loss_sum = 0.0
        total_decays = 0

        while step < training_iters:

            # Load a batch of training data
            images_batch, labels_batch = loader_train.next_batch(batch_size)
            
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), flush=True)

                # Calculate batch loss and accuracy on training set
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., lr_placeholder: current_lr, train_phase: False}) 
                print("-Iter " + str(step) + ", Training Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5), flush=True)

                # Calculate batch loss and accuracy on validation set
                images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
                l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1., lr_placeholder: current_lr, train_phase: False}) 
                val_loss_sum += l
                print("-Iter " + str(step) + ", Validation Loss= " + \
                      "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                      "{:.4f}".format(acc1) + ", Top5 = " + \
                      "{:.4f}".format(acc5), flush=True)
            
            # Run optimization op (backprop)
            sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout, lr_placeholder: current_lr, train_phase: True})
            
            step += 1
            
            # End of epoch for model
            if step % step_save == 0:
                # Save model
                saver.save(sess, path_save, global_step=step)
                print("Model saved at Iter %d !" %(step), flush=True)

                # Check if learning rate should be decayed
                if val_loss_sum >= best_val_loss:
                    print("Validation loss sum did not improve (from %.6f to %.6f)" % (best_val_loss, val_loss_sum), flush=True)
                    current_lr /= lr_decay_factor
                    total_decays += 1
                    if total_decays > max_decays:
                        print("Exceeded maximum number of learning rate decays (%d); stopping early" % max_decays, flush=True)
                        break
                    else:
                        print("Decayed learning rate by factor %.2f to %.6f" % (lr_decay_factor, current_lr), flush=True)
                else:
                    print("Validation loss sum improved from %.6f to %.6f" % (best_val_loss, val_loss_sum), flush=True)
                # Reset values for new epoch
                best_val_loss = val_loss_sum
                val_loss_sum = 0.0
            
        print("Optimization Finished!", flush=True)
        

            
    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...', flush=True)
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1., lr_placeholder: current_lr, train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5), flush=True)

    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total), flush=True)
    

    # Evaluate on the whole test set
    print('results on the test set', flush=True)
    num_batch = loader_test.size()//batch_size
    loader_test.reset()
    with open("pred.txt", "w") as f:
        for i in range(num_batch):
            images_batch, labels_batch = loader_test.next_batch(batch_size)
            labels = sess.run(predict, feed_dict={x: images_batch, keep_dropout: 1., lr_placeholder: current_lr, train_phase: False})[1]
            for j in range(len(labels)):
                item = labels[j]
                f.write("test/"+str((i * batch_size) + j + 1).zfill(8)+".jpg "+ " ".join([str(it) for it in item])+"\n")
