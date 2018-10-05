
# coding: utf-8

# In[1]:


import os
import sys

import numpy as np
import tensorflow as tf
import dataIO as d

from tqdm import *
from utils import *
import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle
# %matplotlib inline


# In[2]:


num_epoch = 500 + 1

batch_size = 100
z_size = 200
cube_len = 32
leak_value = 0.2
weights = {}
l = 0.5
initial_lr = .001
obj = 'chair'
obj_ratio = 0.5
save_interval = 10

experiment_name = 'chair-compress-test'

voxnet_save_path = './voxnet/biasfree_19999.cptk'
gan_save_directory = './models/chair-final-3/'

train_sample_directory = './train_sample/' + experiment_name + '/'
model_directory = './models/' + experiment_name + '/'

img_base_directory = './img/'
img_directory = img_base_directory + experiment_name + '/'

img_test_directory = img_directory + 'test/'

pickle_base_directory = './pickle/'
pickle_directory = pickle_base_directory + experiment_name + '/'

if not os.path.exists(pickle_directory):
    os.makedirs(pickle_directory)
    
if not os.path.exists(train_sample_directory):
    os.makedirs(train_sample_directory)
    
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    
if not os.path.exists(img_directory):
    os.makedirs(img_directory)
        
if not os.path.exists(img_test_directory):
    os.makedirs(img_test_directory)


# In[3]:


tf_epoch = tf.placeholder(tf.int32)
lr = tf.train.exponential_decay(
                      initial_lr,
                      tf_epoch,
                      30,
                      0.5,
                      staircase=True)


# In[4]:


def decoder(z, batch_size=batch_size, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]

    with tf.variable_scope("gen", reuse=reuse):
        g_1 = tf.layers.dense(z, 256*2*2*2, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        g_1 = tf.reshape(g_1, (-1, 2,2,2,256))
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
        g_1 = lrelu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,4,4,4,256), strides=strides, padding="SAME")
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = lrelu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,8,8,8,128), strides=strides, padding="SAME")
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = lrelu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,16,16,16,64), strides=strides, padding="SAME")
        g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
        g_4 = lrelu(g_4)
        
        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size,32,32,32,1), strides=strides, padding="SAME")
        g_5 = tf.nn.sigmoid(g_5)

    print (g_1, 'g1')
    print (g_2, 'g2')
    print (g_3, 'g3')
    print (g_4, 'g4')
    print (g_5, 'g5')
    
    return g_5


def encoder(inputs, encoding_size, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]
    with tf.variable_scope("enc", reuse=reuse):
        xavier_init = tf.contrib.layers.xavier_initializer()

        we1 = tf.get_variable("we1", shape=[4, 4, 4, 1, 32], initializer=xavier_init)
        we2 = tf.get_variable("we2", shape=[4, 4, 4, 32, 64], initializer=xavier_init)
        we3 = tf.get_variable("we3", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
        we4 = tf.get_variable("we4", shape=[4, 4, 4, 128, 256], initializer=xavier_init)    
        d_1 = tf.nn.conv3d(inputs, we1, strides=strides, padding="SAME")
        d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)                               
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, we2, strides=strides, padding="SAME") 
        d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        d_2 = lrelu(d_2, leak_value)
        
        d_3 = tf.nn.conv3d(d_2, we3, strides=strides, padding="SAME")  
        d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
        d_3 = lrelu(d_3, leak_value) 

        d_4 = tf.nn.conv3d(d_3, we4, strides=strides, padding="SAME")     
        d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
        d_4 = lrelu(d_4)

        d_5 = tf.contrib.layers.flatten(d_4)
        d_5 = tf.layers.dense(d_5, encoding_size, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        d_5_no_sigmoid = d_5
        d_5 = tf.nn.sigmoid(d_5)

    print (d_1, 'd1')
    print (d_2, 'd2')
    print (d_3, 'd3')
    print (d_4, 'd4')
    print (d_5, 'd5')

    return d_5, d_5_no_sigmoid

def initialiseWeights():

    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 256], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)    
   

    return weights


# In[5]:


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def voxnet(input_models, phase_train=True, reuse=False):
    strides    = [2,2,2]
    with tf.variable_scope("vox", reuse=reuse):
        vox_1 = tf.layers.conv3d(input_models, 32, [5, 5, 5], strides=strides, padding="valid", trainable = False, use_bias=False, name='vox1')
        vox_1 = tf.layers.batch_normalization(vox_1, training=phase_train, trainable = False, name='vox_bn1')                               
        vox_1 = lrelu(vox_1, leak_value)
        
        vox_2 = tf.layers.conv3d(vox_1, 32, [3, 3, 3], strides=[1,1,1], padding="valid", trainable = False, use_bias=False, name='vox2')
        vox_2 = tf.layers.batch_normalization(vox_2, training=phase_train, trainable = False, name='vox_bn2')                               
        vox_2 = lrelu(vox_2, leak_value)
        vox_2 = tf.layers.max_pooling3d(vox_2, [2,2,2], strides=[1,1,1], padding='valid', name='max_pool1')
        
        vox_flat = tf.contrib.layers.flatten(vox_2)
        vox_3 = tf.layers.dense(vox_flat, 5, kernel_initializer=tf.random_normal_initializer(stddev=0.02), trainable = False, name='vox3')
        vox_3_no_softmax = vox_3
        vox_3 = tf.nn.softmax(vox_3_no_softmax)

    #return 2nd layer of voxnet
    return vox_2


# In[6]:


def plot_voxel(voxel, voxel2=None, saveas=None):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.voxels(voxel.squeeze()>.5, facecolors='red', edgecolors='k')
    ax.set_axis_off()

    if voxel2 is not None:
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.voxels(voxel2.squeeze()>.5, facecolors='red', edgecolors='k')
        ax.set_axis_off()
        
    if saveas is not None:
        plt.savefig(saveas)

#     plt.show()


# In[7]:


initialiseWeights()
x_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1],dtype=tf.float32) 
is_training = tf.placeholder(tf.bool)
#encoding from 32*32*32 down to 200 ::: 163x
enc,_ = encoder(x_vector, z_size, phase_train=is_training, reuse=False)
dec = decoder(enc, phase_train=is_training, reuse=False) 

input_vox = voxnet(x_vector, phase_train=is_training, reuse=False)
decoded_vox = voxnet(dec, phase_train=is_training,reuse=True)

num_voxnet_intermediate_units = np.product(input_vox.get_shape().as_list())

loss_pixel = tf.nn.l2_loss(x_vector - dec) / (batch_size*cube_len*cube_len*cube_len)
loss_perceptual = tf.nn.l2_loss(input_vox - decoded_vox) / num_voxnet_intermediate_units
# weighted sum of the two losses
loss = l*loss_pixel + (1-l)*loss_perceptual


# In[8]:


encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='enc')
#build a list of variable to load
generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')
generator_vars.extend(list(weights.values()))
# train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=encoder_vars+generator_vars)


# In[9]:


extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


# In[10]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())        
all_saver = tf.train.Saver()

gen_loader = tf.train.Saver(generator_vars)
gen_loader.restore(sess, tf.train.latest_checkpoint(gan_save_directory))

vox_loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vox'))
vox_loader.restore(sess, voxnet_save_path)

# load from checkpoint
# all_saver.restore(sess, tf.train.latest_checkpoint(model_directory))


# In[11]:


volumes = d.getAll(obj=obj, train=True, is_local=False, obj_ratio=obj_ratio, cube_len=32)
print ('Using ' + obj + ' Data')
volumes = volumes[...,np.newaxis].astype(np.float)

num_samples = len(volumes)

num_batches = num_samples // batch_size

sess.run(tf.global_variables_initializer())        


# In[ ]:


volumes_test = d.getAll(obj=obj, train=False, is_local=False, obj_ratio=obj_ratio, cube_len=32)
print ('Using ' + obj + ' Data')
volumes_test = volumes_test[...,np.newaxis].astype(np.float)

num_test = len(volumes_test)

num_test_batches = num_test // batch_size


# In[ ]:


train_stat = np.array([])
test_stat = np.array([])

for epoch in range(num_epoch):
    print('train')
    for i in range(num_batches):
        np.random.shuffle(volumes)
        input_voxels = volumes[i*batch_size: (i+1)*batch_size]
        pix_loss, perc_loss, tot_loss, _ = sess.run([loss_pixel, loss_perceptual, loss, train_op], feed_dict={x_vector: input_voxels, tf_epoch: epoch, is_training:True})
        
        train_stat = np.concatenate([train_stat,[tot_loss]])

        print("epoch {}, batch {}: pix loss {:10.5f}, perc loss {:10.5f}, total loss {:10.5f}".format(epoch, i, pix_loss, perc_loss, tot_loss))
        if i==num_batches - 1 and epoch % save_interval == 0:
            # generate and visualize generated images
            recon_voxel = sess.run(dec, feed_dict={x_vector: input_voxels, is_training:True})
            save_sample = np.concatenate((input_voxels, recon_voxel), axis=0)
            
            save_sample.dump("{}{}".format(train_sample_directory, epoch))
            all_saver.save(sess, save_path = '{}compression_gen_{}.ckpt'.format(model_directory, epoch))
            
            for n in range(3):
                plot_voxel(save_sample[n], save_sample[n + batch_size], saveas='{}{}_{}.png'.format(img_directory, epoch, n))
                
            train_stat.dump("{}train_{}.pickle".format(pickle_directory, epoch))
    
    print('validation')
    for i in range(num_test_batches):
        np.random.shuffle(volumes_test)
        input_voxels = volumes_test[i*batch_size: (i+1)*batch_size]
        pix_loss, perc_loss, tot_loss = sess.run([loss_pixel, loss_perceptual, loss], feed_dict={x_vector: input_voxels, tf_epoch: epoch, is_training:False})

        test_stat = np.concatenate([test_stat,[tot_loss]])


        print("epoch {}, batch {}: pix loss {:10.5f}, perc loss {:10.5f}, total loss {:10.5f}".format(epoch, i, pix_loss, perc_loss, tot_loss))
        # generate and visualize generated images
        if i==num_test_batches - 1 and epoch % save_interval == 0:
            recon_voxel = sess.run(dec, feed_dict={x_vector: input_voxels, is_training:False})
            save_sample = np.concatenate((input_voxels, recon_voxel), axis=0)
            for n in range(3):
                plot_voxel(save_sample[n], save_sample[n + batch_size], saveas='{}{}_{}.png'.format(img_test_directory, epoch, n))
            test_stat.dump("{}test_{}.pickle".format(pickle_directory, epoch))

