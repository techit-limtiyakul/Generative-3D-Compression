
# coding: utf-8

# In[1]:


import os
import sys
# import visdom

import numpy as np
import tensorflow as tf
import dataIO as d

from tqdm import *
from utils import *
import pickle

leak_value = 0.2
cube_len   = 32
obj_ratio  = 0.7
obj        = 'vase' 
is_local = False


experiment_name = 'cls_vox'

train_sample_directory = './train_sample/' + experiment_name + '/'
model_directory = './models/' + experiment_name + '/'
img_base_directory = './img/'
img_directory = img_base_directory + experiment_name + '/'
pickle_base_directory = './pickle/'
pickle_directory = pickle_base_directory + experiment_name + '/'
is_local = False

    
weights = {}

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def voxnet(input_models, phase_train=True, reuse=False):
    strides    = [2,2,2]
    with tf.variable_scope("vox", reuse=reuse):
        vox_1 = tf.layers.conv3d(input_models, 32, [5, 5, 5], strides=strides, padding="valid", use_bias=False, name='vox1')
        vox_1 = tf.layers.batch_normalization(vox_1, training=phase_train, name='vox_bn1')                               
        vox_1 = lrelu(vox_1, leak_value)
        
        vox_2 = tf.layers.conv3d(vox_1, 32, [3, 3, 3], strides=[1,1,1], padding="valid", use_bias=False, name='vox2')
        vox_2 = tf.layers.batch_normalization(vox_2, training=phase_train, name='vox_bn2')                               
        vox_2 = lrelu(vox_2, leak_value)
        vox_2 = tf.layers.max_pooling3d(vox_2, [2,2,2], strides=[1,1,1], padding='valid', name='max_pool1')
        
        vox_flat = tf.contrib.layers.flatten(vox_2)
        vox_3 = tf.layers.dense(vox_flat, 5, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='vox3')
        vox_3_no_softmax = vox_3
        vox_3 = tf.nn.softmax(vox_3_no_softmax)


    return vox_3, vox_3_no_softmax

    


# In[2]:


batch_size = 32
beta = 0.9
cube_len = 32
def train(checkpoint=None):
    x = tf.placeholder(shape=[None,cube_len,cube_len,cube_len,1],dtype=tf.float32) 
    label = tf.placeholder(shape=[None,5],dtype=tf.float32) 
    out, out_no_softmax = voxnet(x, reuse=False)
    #a = tf.argmax(out_no_softmax, axis=1)
    #b = tf.argmax(label, axis=1)
    n_p = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(out_no_softmax, axis=1), tf.argmax(label, axis=1)), tf.float32))/batch_size
    
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits (logits=out_no_softmax, labels=label))
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=0.001,beta1=beta).minimize(loss)
    
    saver = tf.train.Saver() 
    d_losses = []
    g_losses = []
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())   
        
        volumes0 = d.getAll(obj='vase', train=True, is_local=is_local, obj_ratio=obj_ratio, cube_len=32)
        volumes1 = d.getAll(obj='cup', train=True, is_local=is_local, obj_ratio=1.0, cube_len=32)
        volumes2 = d.getAll(obj='bottle', train=True, is_local=is_local, obj_ratio=obj_ratio, cube_len=32)
        volumes3 = d.getAll(obj='chair', train=True, is_local=is_local, obj_ratio=0.25, cube_len=32)
        volumes4 = d.getAll(obj='bowl', train=True, is_local=is_local, obj_ratio=1.0, cube_len=32)
        volumes = np.append(np.append(np.append(np.append(volumes0, volumes1, axis=0), volumes2, axis=0), volumes3, axis=0),volumes4, axis=0)
        print(volumes0.shape)
        labels0 = np.full((volumes0.shape[0], 1), 0)
        labels1 = np.full((volumes1.shape[0], 1), 1)
        labels2 = np.full((volumes2.shape[0], 1), 2)
        labels3 = np.full((volumes3.shape[0], 1), 3)
        labels4 = np.full((volumes4.shape[0], 1), 4)
        #print(labels0.shape)
        #print(labels1.shape)
        #print(labels2.shape)
        #print(labels3.shape)
        #print(labels4.shape)
        labels = np.append(np.append(np.append(np.append(labels0, labels1, axis=0), labels2, axis=0), labels3, axis=0), labels4, axis=0)
        #print(labels.shape)
        #print ('Using ' + obj + ' Data')
        volumes = volumes[...,np.newaxis].astype(np.float32)

        #s = np.arange(volumes.shape[0])
        #np.random.shuffle(s)
        #volumes = volumes[s]
        #labels = labels[s]
        #print(labels[:20])
        #print(labels.shape)
        #print(volumes.shape)
        #print(labels[:20])
        #print(np.sum(labels==4))
        #print(labels0)
        #print(volumes.shape)
        ###
        for epoch in range(20000):
            idx = np.random.randint(len(volumes), size=batch_size)
            input_x = volumes[idx]
            lab = labels[idx, 0]
            #print( 'Hiiiiiii')
            n_values = 5
            lab = np.eye(n_values)[lab]
            #print(lab.shape)
            lab = lab.astype(np.float32)
            #print(type(lab[20,3]))
            #print(type( input_x[0,0,0,0,0]))
            #print( input_x.shape)
            #for i in lab:
            #    temp = np.zeros()
            _, l, acc = sess.run([optimizer_op_d, loss, n_p],feed_dict={label:lab, x: input_x})
            d_losses.append(l)
            #print(m)
            #print(n)
            print ('Discriminator Training ', "epoch: ",epoch, " loss: ", l, "accuracy: ", acc)
            
            if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
            
            if not os.path.exists(pickle_directory):
                    os.makedirs(pickle_directory)
                    
            if epoch%100==0:
                with open(pickle_directory + 'd_loss.pickle', 'wb') as file:
                    pickle.dump(l, file)
            if epoch % 1000 == 999:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)      
                saver.save(sess, save_path = model_directory + '/biasfree_' + str(epoch) + '.cptk')
                
            


# In[3]:


train()

