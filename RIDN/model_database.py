import tensorflow as tf
import numpy as np

def subpixel_LR2HR_ZeroCenter(image_LR):
    img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
    img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
    img_HR3 = tf.depth_to_space(image_LR[:,:,:,8:12], 2)
    return tf.concat([img_HR1+0.4488*255, img_HR2+0.4371*255, img_HR3+0.4040*255], 3)

def subpixel_HR2LR_ZeroCenter(image_HR):
    img_LR1 = tf.space_to_depth(image_HR[:,:,:,0:1], 2)
    img_LR2 = tf.space_to_depth(image_HR[:,:,:,1:2], 2)
    img_LR3 = tf.space_to_depth(image_HR[:,:,:,2:3], 2)
    return tf.concat([img_LR1-0.4488*255, img_LR2-0.4371*255, img_LR3-0.4040*255], 3)

def CIR_RIDN(input, is_training=True):
    num_feature = 64
    R4G4B4 = subpixel_HR2LR_ZeroCenter(input)
    with tf.variable_scope('stage1'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(R4G4B4, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, epsilon=1e-5, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')#batch*50*50*12
            S1InterG = R4G4B4[:,:,:,4:8] + output #batch*50*50*12
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
    return subpixel_LR2HR_ZeroCenter(S3RGB + output)