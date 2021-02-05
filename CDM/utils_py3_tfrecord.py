import gc
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from msssim import MultiScaleSSIM

def load_images(filelist):
    if not isinstance(filelist, list):
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file)
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data

def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath)

def imcpsnr(im1, im2, peak=255, b=0):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1[b:im1.shape[0]-b,b:im1.shape[1]-b,:]
    im2 = im2[b:im2.shape[0]-b,b:im2.shape[1]-b,:]
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    return 10 * np.log10(255 ** 2 / mse)

def impsnr(im1, im2, peak=255, b=0):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    im1 = im1[b:im1.shape[0]-b,b:im1.shape[1]-b,:]
    im2 = im2[b:im2.shape[0]-b,b:im2.shape[1]-b,:]
    csnr = np.ones(im1.shape[2])
    for i in range(im1.shape[2]):
        mse = ((im1[:,:,i].astype(np.float) - im2[:,:,i].astype(np.float)) ** 2).mean()
        csnr[i] = 10 * np.log10(255 ** 2 / mse)
    return csnr

def getBayer(RGB_Image):
    RGB_Image = np.squeeze(RGB_Image)
    h, w, _ = RGB_Image.shape
    Bayer = np.zeros([h, w], dtype=np.uint8)
    Bayer[0:h:2,0:w:2] = RGB_Image[0:h:2,0:w:2,0]#R
    Bayer[1:h:2,0:w:2] = RGB_Image[1:h:2,0:w:2,1]#G1
    Bayer[0:h:2,1:w:2] = RGB_Image[0:h:2,1:w:2,1]#G2
    Bayer[1:h:2,1:w:2] = RGB_Image[1:h:2,1:w:2,2]#B
    return Bayer

def MS_SSIM(im1, im2, b=0):
    im1 = im1[:,b:im1.shape[1]-b,b:im1.shape[2]-b,:]
    im2 = im2[:,b:im2.shape[1]-b,b:im2.shape[2]-b,:]
    return MultiScaleSSIM(im1, im2)

def SSIM(im1, im2, b=0):
    im1 = tf.convert_to_tensor(im1[:,b:im1.shape[1]-b,b:im1.shape[2]-b,:], dtype=np.float32)
    im2 = tf.convert_to_tensor(im2[:,b:im2.shape[1]-b,b:im2.shape[2]-b,:], dtype=np.float32)
    return tf.image.ssim(im1, im2, max_val=255)
