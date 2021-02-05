import tensorflow as tf
def subpixel_LR2HR(image_LR):
    c = image_LR.shape[3]
    if c==4:
        img_HR = tf.depth_to_space(image_LR, 2)
        return img_HR
    elif c==8:
        img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
        img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
        return tf.concat([img_HR1, img_HR2], 3)
    elif c==12:
        img_HR1 = tf.depth_to_space(image_LR[:,:,:,0:4], 2)
        img_HR2 = tf.depth_to_space(image_LR[:,:,:,4:8], 2)
        img_HR3 = tf.depth_to_space(image_LR[:,:,:,8:12], 2)
        return tf.concat([img_HR1, img_HR2, img_HR3], 3)
    else:
        print('ERROR!')

def CIR_CDM(input, is_training=True):
    num_feature = 64
    with tf.variable_scope('Pre_processing'):
        _, h, w, _ = input.shape
        sub_R  = input[:, 0:h:2, 0:w:2, :]
        sub_G1 = input[:, 0:h:2, 1:w:2, :]
        sub_G2 = input[:, 1:h:2, 0:w:2, :]
        sub_B  = input[:, 1:h:2, 1:w:2, :]
        RGGB   = tf.concat([sub_R,sub_G1,sub_G2,sub_B], 3)
        R4     = tf.concat([sub_R,sub_R,sub_R,sub_R], 3)
        G4     = tf.concat([sub_G1,sub_G1,sub_G2,sub_G2], 3)
        B4     = tf.concat([sub_B,sub_B,sub_B,sub_B], 3)
        R4G4B4 = tf.concat([sub_R,sub_R,sub_R,sub_R,sub_G1,sub_G1,sub_G2,sub_G2,sub_B,sub_B,sub_B,sub_B], 3)
    with tf.variable_scope('stage1'):
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(RGGB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 12, 3, padding='same')#batch*50*50*12
            S1InterRGB = R4G4B4 + output #batch*50*50*12
            S1InterR = S1InterRGB[:,:,:,0:4]
            S1InterG = S1InterRGB[:,:,:,4:8]
            S1InterB = S1InterRGB[:,:,:,8:12]
    with tf.variable_scope('stage2-1'):
        S2RG = tf.concat([S1InterR, S1InterG], 3)
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(S2RG, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')
            S2InterR = S1InterR + output
    with tf.variable_scope('stage2-2'):
        S2GB = tf.concat([S1InterG, S1InterB], 3)
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(S2GB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 4, 3, padding='same')
            S2InterB = S1InterB + output
    with tf.variable_scope('stage3'):
        S3RGB = tf.concat([S2InterR, S1InterG, S2InterB], 3)
        with tf.variable_scope('block1'):
            output = tf.layers.conv2d(S3RGB, num_feature, 3, padding='same', activation=tf.nn.relu)
        for layers in range(2, 9 + 1):
            with tf.variable_scope('block%d' % layers):
                output = tf.layers.conv2d(output, num_feature, 3, padding='same', name='conv%d' % layers)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
        with tf.variable_scope('block10'):
            output = tf.layers.conv2d(output, 12, 3, padding='same')
    return subpixel_LR2HR(S3RGB+output)
