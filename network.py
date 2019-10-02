from tensorflow.keras import models, layers

def conv_2d_block(input_layer, n_filter, ksize, padding='same', activation='relu', name='block'):
    output = layers.Conv2D(n_filter, ksize, padding=padding, name=name+'_conv')(input_layer)
    if activation=='leaky_relu':
        output = layers.LeakyReLU(0.01, name=name+'_act')(output)
    else:
        output = layers.Activation('relu', name=name+'_act')(output)
    return output

def upconv_2d_block(input_layer, n_filter, ksize, strides, padding='same', activation='relu', name='block'):
    output = layers.Conv2DTranspose(n_filter, ksize, strides, padding=padding, name=name+'_conv')(input_layer)
    if activation=='leaky_relu':
        output = layers.LeakyReLU(0.01, name=name+'_act')(output)
    else:
        output = layers.Activation('relu', name=name+'_act')(output)
    return output

def residual_2d_block(input_layer, n_filter, ksize, padding='same', activation='relu', mode=None, name='block'):
    '''
    Reference : https://arxiv.org/pdf/1612.02177.pdf
    '''
    if mode == "up":
        output = upconv_2d_block(input_layer, n_filter, (1, 6), strides=(1, 6), padding='same', 
                         activation=activation, name=name+'_up')
    else:
        output = conv_2d_block(input_layer, n_filter, ksize, activation=activation, name=name+'_conv1')
    output = conv_2d_block(output, n_filter, ksize, activation='linear', name=name+'_conv2')
    output = layers.Add(name=name+'_add')([output, input_layer])
    
    return output

def conv_3d_block(input_layer, n_filter, ksize, padding='same', activation='relu', name='block'):
    output = layers.Conv3D(n_filter, ksize, padding=padding, name=name+'_conv')(input_layer)
    if activation=='leaky_relu':
        output = layers.LeakyReLU(0.01, name=name+'_act')(output)
    else:
        output = layers.Activation('relu', name=name+'_act')(output)
    return output

def upconv_3d_block(input_layer, n_filter, ksize, strides, padding='same', activation='relu', name='block'):
    output = layers.Conv3DTranspose(n_filter, ksize, strides, padding=padding, name=name+'_conv')(input_layer)
    if activation=='leaky_relu':
        output = layers.LeakyReLU(0.01, name=name+'_act')(output)
    else:
        output = layers.Activation('relu', name=name+'_act')(output)
    return output

def residual_3d_block(input_layer, n_filter, ksize, padding='same', activation='relu', mode=None, name='block'):
    '''
    Reference : https://arxiv.org/pdf/1612.02177.pdf
    '''
    if mode == "up":
        output = upconv_3d_block(input_layer, n_filter, (1, 1, 6), strides=(1, 1, 6), padding='same', 
                         activation=activation, name=name+'_up')
    else:
        output = conv_3d_block(input_layer, n_filter, ksize, activation=activation, name=name+'_conv1')
    output = conv_3d_block(output, n_filter, ksize, activation='linear', name=name+'_conv2')
    output = layers.Add(name=name+'_add')([output, input_layer])
    
    return output



def SR3D(input_shape=(None, None, None, 1), layer_activation='relu', last_activation='linear', name='3D_SR'):
    
    '''
    input_shape : (Cor, Sag, Axi)
    '''
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_3d_block(input_layer, 64, 3, activation=layer_activation, name=name+'_en1')
    
    for_concat = layers.UpSampling3D(size=(1,1,6), name=name+'_up_en1')(en1)
    
    en2 = conv_3d_block(en1, 128, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = conv_3d_block(en2, 256, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = conv_3d_block(en3, 512, 3, activation=layer_activation, name=name+'_en4')
    
    up = upconv_3d_block(en4, 64, (1, 1, 6), strides=(1, 1, 6), padding='same', 
                         activation=layer_activation, name=name+'_up')
    concat = layers.Concatenate(axis=-1, name=name+'_concat')([for_concat, up])
    
    refine = conv_3d_block(concat, 64, 3, activation=last_activation, name=name+'_refine')
    
    output = conv_3d_block(refine, 1, 1, activation=last_activation, name=name+'_output')
    
    return models.Model(inputs=input_layer, outputs=output, name=name)






def SR3D_res(input_shape=(None, None, None, 1), residual_channel=64, layer_activation='relu', last_activation='linear', name='3D_SR'):
    
    '''
    input_shape : (Cor, Sag, Axi)
    '''
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_3d_block(input_layer, residual_channel, 3, activation=layer_activation, name=name+'_en1')
    
    for_concat = layers.UpSampling3D(size=(1,1,6), name=name+'_up_en1')(en1)
    
    en2 = residual_3d_block(en1, residual_channel, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = residual_3d_block(en2, residual_channel, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = residual_3d_block(en3, residual_channel, 3, activation=layer_activation, name=name+'_en4')
    
    up = upconv_3d_block(en4, residual_channel, (1, 1, 6), strides=(1, 1, 6), padding='same', 
                         activation=layer_activation, name=name+'_up')
    
    concat = layers.Concatenate(axis=-1, name=name+'_concat')([for_concat, up])
    
    refine = conv_3d_block(concat, residual_channel, 3, activation=last_activation, name=name+'_refine')
    
    output = conv_3d_block(refine, 1, 1, activation=last_activation, name=name+'_output')
    
    return models.Model(inputs=input_layer, outputs=output, name=name)


def dis3D(input_shape=(None, None, None, 1), layer_activation='relu', last_activation='sigmoid', name='3D_dis'):
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_3d_block(input_layer, 64, 3, activation=layer_activation, name=name+'_en1')
    
    en2 = layers.MaxPool3D(name=name+'_pool1')(en1)
    en2 = conv_3d_block(en1, 128, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = layers.MaxPool3D(name=name+'_pool2')(en2)
    en3 = conv_3d_block(en2, 256, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = layers.MaxPool3D(name=name+'_pool3')(en3)
    en4 = conv_3d_block(en3, 512, 3, activation=layer_activation, name=name+'_en4')
    
    GAP = layers.GlobalAvgPool3D(name=name+'_gap')(en4)
    
    output = layers.Dense(1, activation=last_activation)(GAP)
    
    return models.Model(inputs=input_layer, outputs=output, name=name)

def dis3D_res(input_shape=(None, None, None, 1), residual_channel=64, layer_activation='relu', last_activation='sigmoid', name='3D_dis'):
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_3d_block(input_layer, residual_channel, 3, activation=layer_activation, name=name+'_en1')
    
    en2 = layers.MaxPool3D(name=name+'_pool1')(en1)
    en2 = residual_3d_block(en1, residual_channel, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = layers.MaxPool3D(name=name+'_pool2')(en2)
    en3 = residual_3d_block(en2, residual_channel, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = layers.MaxPool3D(name=name+'_pool3')(en3)
    en4 = residual_3d_block(en3, residual_channel, 3, activation=layer_activation, name=name+'_en4')
    
    GAP = layers.GlobalAvgPool3D(name=name+'_gap')(en4)
    
    output = layers.Dense(1, activation=last_activation)(GAP)
    
    return models.Model(inputs=input_layer, outputs=output, name=name)

def dis2D(input_shape=(None, None, 1), layer_activation='relu', last_activation='sigmoid', name='2D_dis'):
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_2d_block(input_layer, 64, 3, activation=layer_activation, name=name+'_en1')
    
    en2 = layers.MaxPool2D(name=name+'_pool1')(en1)
    en2 = conv_2d_block(en1, 128, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = layers.MaxPool2D(name=name+'_pool2')(en2)
    en3 = conv_2d_block(en2, 256, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = layers.MaxPool2D(name=name+'_pool3')(en3)
    en4 = conv_2d_block(en3, 512, 3, activation=layer_activation, name=name+'_en4')
    
    GAP = layers.GlobalAvgPool2D(name=name+'_gap')(en4)
    
    output = layers.Dense(1, activation=last_activation)(GAP)
    
    return models.Model(inputs=input_layer, outputs=output, name=name)

def dis2D_res(input_shape=(None, None, 1), residual_channel=64, layer_activation='relu', last_activation='sigmoid', name='2D_dis'):
    
    input_layer = layers.Input(input_shape, name=name+'_input')
    
    en1 = conv_2d_block(input_layer, residual_channel, 3, activation=layer_activation, name=name+'_en1')
    
    en2 = layers.MaxPool2D(name=name+'_pool1')(en1)
    en2 = residual_2d_block(en1, residual_channel, 3, activation=layer_activation, name=name+'_en2')
    
    en3 = layers.MaxPool2D(name=name+'_pool2')(en2)
    en3 = residual_2d_block(en2, residual_channel, 3, activation=layer_activation, name=name+'_en3')
    
    en4 = layers.MaxPool2D(name=name+'_pool3')(en3)
    en4 = residual_2d_block(en3, residual_channel, 3, activation=layer_activation, name=name+'_en4')
    
    GAP = layers.GlobalAvgPool2D(name=name+'_gap')(en4)
    
    output = layers.Dense(1, activation=last_activation)(GAP)
    
    return models.Model(inputs=input_layer, outputs=output, name=name)