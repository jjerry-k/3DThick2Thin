import tensorflow as tf
from tensorflow.keras import losses

def mse(y_true, y_pred):
    output = tf.reduce_mean(losses.mean_squared_error(y_true, y_pred))
    return output

def gradient_3d_loss(x, y):    
    x_cen = x[:, 1:-1, 1:-1, 1:-1]
    x_shape = tf.shape(x)
    grad_x = tf.zeros_like(x_cen)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = tf.slice(x, [0, i+1, j+1, k+1, 0], [x_shape[0], x_shape[1]-2, x_shape[2]-2, x_shape[3]-2, x_shape[4]])
                if i*i + j*j + k*k == 0:
                    temp = tf.zeros_like(x_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(i*i + j*j + k*k, tf.float32)), tf.nn.relu(x_slice - x_cen))
                grad_x = grad_x + temp

    y_cen = y[:, 1:-1, 1:-1, 1:-1]
    y_shape = tf.shape(y)
    grad_y = tf.zeros_like(y_cen)
    for ii in range(-1, 2):
        for jj in range(-1, 2):
            for kk in range(-1, 2):
                y_slice = tf.slice(y, [0, ii + 1, jj + 1, kk + 1, 0], [y_shape[0], y_shape[1] - 2, y_shape[2] - 2, y_shape[3] - 2, y_shape[4]])
                if ii*ii + jj*jj + kk*kk== 0:
                    temp = tf.zeros_like(y_cen)
                else:
                    temp = tf.scalar_mul(1.0 / tf.sqrt(tf.cast(ii * ii + jj * jj + kk * kk, tf.float32)), tf.nn.relu(y_slice - y_cen))
                grad_y = grad_y + temp

    output = tf.square(grad_x - grad_y)
    output = tf.reduce_mean(output)
    return output

def mse_grad_loss(y_true, y_pred):
    mse_loss = mse(y_true, y_pred)
    grad_loss = gradient_3d_loss(y_true, y_pred)
    alpha = 1/7.85
    return 1.*mse_loss + alpha*grad_loss
    
    
def tf_joint_histogram(y_true, y_pred):
    """
    y_true : [batch, Cor, Sag, Axi, 1]
    y_pred : [batch, Cor, Sag, Axi, 1]
    """
    #print("joint1")
    vmax = 255
    #b, h, w, c = tf.shape(y_true)
    
    
    # Intensity Scaling ( [batch, Cor, Sag, Axi, 1] -> [batch, Cor, Sag, Axi] )
    max_true_int = tf.reduce_max(y_true, axis = [1,2,3], keepdims=True)
    max_pred_int = tf.reduce_max(y_pred, axis = [1,2,3], keepdims=True)
    tmp_true = tf.squeeze(tf.round(y_true / max_true_int * vmax), axis=-1)
    tmp_pred = tf.squeeze(tf.round(y_pred / max_pred_int * vmax), axis=-1)
    
    #print("joint2")
    # [batch, height, width, channel] -> [batch, height * width, channel] -> [batch, channel, height * width]
    flat_true = tf.transpose(tf.reshape(tmp_true,[tf.shape(y_true)[0], tf.shape(y_true)[1]*tf.shape(y_true)[2], tf.shape(y_true)[3]]), [0, 2, 1])
    # [batch, channel, height * width] -> [batch * channel, height * width]
    flat_true = tf.reshape(flat_true, [tf.shape(y_true)[0]*tf.shape(y_true)[3], tf.shape(y_true)[1]*tf.shape(y_true)[2]])
    # [batch, height, width, channel] -> [batch, height * width, channel] -> [batch, channel, height * width]
    flat_pred = tf.transpose(tf.reshape(tmp_pred, [tf.shape(y_true)[0], tf.shape(y_true)[1]*tf.shape(y_true)[2], tf.shape(y_true)[3]]), [0, 2, 1])
    # [batch, channel, height * width] -> [batch * channel, height * width]
    flat_pred = tf.reshape(flat_pred, [tf.shape(y_true)[0]*tf.shape(y_true)[3], tf.shape(y_true)[1]*tf.shape(y_true)[2]])
    
    # [b*c, 65536]
    output = (flat_pred * (vmax+1)) + (flat_true+1)
    
    output = tf.map_fn(lambda x : tf.cast(tf.histogram_fixed_width(x, value_range=[1, (vmax+1)**2], nbins=(vmax+1)**2), 'float32'), output)
    # [b, c, 256, 256] -> [b, 256, 256, c]
    output = tf.transpose(tf.reshape(output, [tf.shape(y_true)[0], tf.shape(y_true)[3], vmax+1, vmax+1]), [0, 2, 3, 1])
    #print("joint5")
    return output, y_true, y_pred

def mutual_information_single(hist2d):
    tmp = tf.cast(hist2d, dtype='float64')
    pxy = tmp / tf.reduce_sum(tmp)
    px = tf.reduce_sum(pxy, axis=1)
    py = tf.reduce_sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = tf.greater(pxy, 0)
    return tf.reduce_sum(tf.boolean_mask(pxy, nzs) * tf.math.log(tf.boolean_mask(pxy, nzs) / tf.boolean_mask(px_py, nzs)))


def mutual_information(y_true, y_pred):
    """
    y_true : [batch, height, width, channel]
    y_pred : [batch, height, width, channel]
    """
    # [b, 256, 256, c]
    joint_histogram, _, _ = tf_joint_histogram(y_true, y_pred)
    #b, h, w, c = tf.shape(joint_histogram)
    #print("mutual1")
    # [b*c, 256, 256]
    reshape_joint_histogram = tf.reshape(tf.transpose(joint_histogram, [0, 3, 1, 2]), [tf.shape(joint_histogram)[0]*tf.shape(joint_histogram)[-1], tf.shape(joint_histogram)[1], tf.shape(joint_histogram)[2]])
    #print("mutual2")
    output = tf.map_fn(lambda x : mutual_information_single(x), reshape_joint_histogram, dtype=tf.float64)
    #print("mutual3")
    output = tf.reshape(output, [tf.shape(joint_histogram)[0], tf.shape(joint_histogram)[-1]])
    return tf.cast( - tf.reduce_mean(output, axis=1), 'float32')

    
def mse_mi(y_true, y_pred):
    mse_loss = mse(y_true, y_pred)
    mi = mutual_information(y_true, y_pred)
    alpha = 1000
    return 1.*mse_loss + alpha*mi   