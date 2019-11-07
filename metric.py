import numpy as np
import tensorflow as tf
from tensorflow import nn

def psnr(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    global_max = tf.maximum(tf.cast(tf.reduce_max(y_true), tf.float64), tf.reduce_max(y_pred))
    
    normed_true = y_true / global_max
    normed_pred = y_pred / global_max
    
    dim = tf.rank(normed_true)
    reduction_dim = tf.range(-(dim-1), 0, 1)

    mse = tf.reduce_mean(tf.squared_difference(normed_true, normed_pred), axis= reduction_dim)
    result = tf.subtract(20. * tf.log(global_max) / tf.cast(tf.log(10.0), tf.float64),
        np.float64(10 / np.log(10)) * tf.log(mse))
    return result


def gaussian(kernel_size, sigma):
    gauss = tf.exp(([-(x - kernel_size//2)**2/float(2*sigma**2) for x in range(kernel_size)]))
    return gauss/tf.reduce_sum(gauss)

def create_kernel(kernel_size, sigma, dim='2d'):
    if dim == '2d':
        _1D_gauss = gaussian(kernel_size, sigma)[..., tf.newaxis]
        _2D_gauss = (_1D_gauss * tf.transpose(_1D_gauss, (1,0)))
        return _2D_gauss[..., tf.newaxis, tf.newaxis]
    else:
        _1D_gauss = gaussian(kernel_size, sigma)[..., tf.newaxis, tf.newaxis]
        _3D_gauss = (_1D_gauss * tf.transpose(_1D_gauss, (1,0,2)) * tf.transpose(_1D_gauss, (1,2,0)))
        return _3D_gauss[..., tf.newaxis, tf.newaxis]

    
class ssim():
    def __init__(self, dim):
        self.dim = dim
        self.conv = {'2d': nn.conv2d, 
                     '3d': nn.conv3d}
        
        self.strides = {'2d': [1,1,1,1], 
                        '3d': [1,1,1,1,1]}
        
    def run(self, y_true, y_pred):
        global_max = tf.maximum(tf.reduce_max(y_true), tf.reduce_max(y_pred))
    
        y_true = tf.cast(y_true/global_max, tf.float32)
        y_pred = tf.cast(y_pred/global_max, tf.float32)

        kernel = create_kernel(11, 1.5, dim=self.dim)

        mu1 = self.conv[self.dim](y_true, kernel, strides=self.strides[self.dim], padding = 'SAME')
        mu2 = self.conv[self.dim](y_pred, kernel, strides=self.strides[self.dim], padding = 'SAME')

        mu1_sq = tf.square(mu1)
        mu2_sq = tf.square(mu2)
        mu1_mu2 = mu1*mu2


        sigma1_sq = self.conv[self.dim](tf.square(y_true), kernel, strides=self.strides[self.dim], padding = 'SAME') - mu1_sq
        sigma2_sq = self.conv[self.dim](tf.square(y_pred), kernel, strides=self.strides[self.dim], padding = 'SAME') - mu2_sq
        sigma12 = self.conv[self.dim](tf.multiply(y_true, y_pred), kernel, strides=self.strides[self.dim], padding = 'SAME') - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        return tf.reduce_mean(ssim_map)