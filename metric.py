import tensorflow as tf

def psnr(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    global_max = tf.maximum(tf.cast(tf.reduce_max(y_true), tf.float64), tf.reduce_max(y_pred))
    
    normed_true = y_true / global_max
    normed_pred = y_pred / global_max
    
    result = tf.image.psnr(normed_true, normed_pred, 1)
    return result