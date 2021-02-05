import tensorflow as tf
import numpy as np
# network structure
def subpixel_LR2HR(image_LR):
    c = image_LR.shape[3]
    if c==4:#64*50*50*4
        img_HR = tf.depth_to_space(image_LR, 2)
        return img_HR
    elif c==8:#64*50*50*8
        img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
        img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
        return tf.concat([img_HR1, img_HR2], 3)
    elif c==12:#64*50*50*12
        img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
        img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
        img_HR3 = tf.depth_to_space(image_LR[:,:,:,8:12], 2)
        return tf.concat([img_HR1, img_HR2, img_HR3], 3)
    else:
        print('ERROR!')

def subpixel_HR2LR_new(image_HR):
    c = image_HR.shape[3]
    if c==1:#64*50*50*4
        img_LR = tf.space_to_depth(image_HR, 2)
        return img_LR
    elif c==2:#64*50*50*8
        img_LR1 = tf.space_to_depth(image_HR[:,:,:,0:1], 2)
        img_LR2 = tf.space_to_depth(image_HR[:,:,:,1:2], 2)
        return tf.concat([img_LR1, img_LR2], 3)
    elif c==3:#64*50*50*12
        img_LR1 = tf.space_to_depth(image_HR[:,:,:,0:1], 2)
        img_LR2 = tf.space_to_depth(image_HR[:,:,:,1:2], 2)
        img_LR3 = tf.space_to_depth(image_HR[:,:,:,2:3], 2)
        return tf.concat([img_LR1, img_LR2, img_LR3], 3)
    else:
        print('ERROR!')

def CIR_CAR(input, is_training=True):
    num_feature = 64
    R4G4B4 = subpixel_HR2LR_new(input)
    with tf.variable_scope('stage1'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(R4G4B4[:,:,:,4:8], num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')
            S1InterG = R4G4B4[:,:,:,4:8] + output 
            S2RG = tf.concat([R4G4B4[:,:,:,0:4], S1InterG], 3)
            S2GB = tf.concat([S1InterG, R4G4B4[:,:,:,8:12]], 3)
    with tf.variable_scope('stage2-1'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(S2RG, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')
            S2InterR = R4G4B4[:,:,:,0:4] + output
    with tf.variable_scope('stage2-2'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(S2GB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')
            S2InterB = R4G4B4[:,:,:,8:12] + output
    with tf.variable_scope('stage3'):
        S3RGB = tf.concat([S2InterR, S1InterG, S2InterB], 3)
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(S3RGB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 12, 3, padding='same')
    return subpixel_LR2HR(S3RGB + output)