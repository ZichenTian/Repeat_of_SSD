import keras.backend as K
import tensorflow as tf

def L2_Normalization(x):
    x = K.l2_normalize(x, axis=1)
    #x *= 20        #20貌似不用乘
    return x        #作者源代码中对conv4_3要进行这一步20×L2_Normalization的操作，用于调整loss的权重

def L2_Normalization_Shape(input_shape):
    return tuple(input_shape)
