from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core  import  Lambda, Reshape
from keras.layers.merge import concatenate
from ssd_utils import L2_Normalization, L2_Normalization_Shape


class SSD300_Param(object):
    def __init__(self, img_shape=(300,300,3), 
                num_classes=21):
        self.img_shape = img_shape
        self.num_classes = num_classes


def SSD300(input_shape=(300,300,3), num_classes=21):
    inputs = x = Input(shape=input_shape, name='inputs')       #input_shape一般是(300,300,3)
    #block1
    #(300,300,3)
    #x = BatchNormalization()(x)
    x = Conv2D(filters = 64,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv1_1')(x)
    x = Conv2D(filters = 64,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv1_2') (x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2),padding='same', 
                    data_format='channels_last', name='pool1')(x)
    #block2
    #(150,150,64)
    x = Conv2D(filters = 128,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv2_1')(x)
    x = Conv2D(filters = 128,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv2_2')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2),padding='same', 
                    data_format='channels_last', name='pool2')(x)
    #block3
    #(75,75,128)
    x = Conv2D(filters = 256,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv3_1')(x)
    x = Conv2D(filters = 256,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv3_2')(x) 
    x = Conv2D(filters = 256,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv3_3')(x) 
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2),padding='same', 
                    data_format='channels_last', name='pool3')(x)
    #block4
    #(38,38,256)
    x = Conv2D(filters = 512,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv4_1')(x)
    x = Conv2D(filters = 512,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv4_2')(x) 
    conv4_3 = x = Conv2D(filters = 512,kernel_size = (3,3), data_format='channels_last',     #conv4_3,(38,38,512)
                activation='relu',padding='same', name='conv4_3')(x) 
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2),padding='same', 
                    data_format='channels_last', name='pool4')(x)
    #block5
    #(19,19,512)
    x = Conv2D(filters = 512,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv5_1')(x)
    x = Conv2D(filters = 512,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv5_2')(x) 
    x = Conv2D(filters = 512,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv5_3')(x) 
    x = MaxPooling2D(pool_size=(3,3), strides=(1,1),padding='same', 
                    data_format='channels_last', name='pool5')(x)
    #block6
    #(19,19,512)
    x = Conv2D(filters = 1024,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='same', name='conv6', dilation_rate=(6,6))(x)
                #Keras高版本貌似取消了AtrousConv2D这个函数，用Conv2D中的dilation_rate来设置膨胀系数 
    #block7
    #(19,19,1024)
    conv7 = x = Conv2D(filters = 1024,kernel_size = (1,1), data_format='channels_last',      #conv7,(19,19,1024)
                activation='relu',padding='same', name='conv7')(x)
    #block8
    #(19,19,1024)
    x = Conv2D(filters = 256,kernel_size = (1,1), data_format='channels_last',
                activation='relu',padding='same', name='conv8_1')(x)
    x = ZeroPadding2D(padding=(1,1), data_format='channels_last', name='padding1')(x)
    conv8_2 = x = Conv2D(filters = 512,kernel_size = (3,3), strides = (2,2), data_format='channels_last',
                activation='relu',padding='valid', name='conv8_2')(x)       #conv8_2,(10,10,512)
    #block9
    #(10,10,512)
    x = Conv2D(filters = 128,kernel_size = (1,1), data_format='channels_last',
                activation='relu',padding='same', name='conv9_1')(x)
    x = ZeroPadding2D(padding=(1,1), data_format='channels_last', name='padding2')(x)
    conv9_2 = x = Conv2D(filters = 256,kernel_size = (3,3), strides = (2,2), data_format='channels_last',
                activation='relu',padding='valid', name='conv9_2')(x)       #conv9_2,(5,5,256)
    #block10
    #(5,5,256)
    x = Conv2D(filters = 128,kernel_size = (1,1), data_format='channels_last',
                activation='relu',padding='same', name='conv10_1')(x)
    conv10_2 = x = Conv2D(filters = 256,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='valid', name='conv10_2')(x)      #conv10_2,(3,3,256)
    #block11
    #(3,3,256)
    x = Conv2D(filters = 128,kernel_size = (1,1), data_format='channels_last',
                activation='relu',padding='same', name='conv11_1')(x)
    conv11_2 = x = Conv2D(filters = 256,kernel_size = (3,3), data_format='channels_last',
                activation='relu',padding='valid', name='conv11_2')(x)      #conv11_2,(1,1,256)
    #(1,1,256)

    #从conv4_3开始的检测与分类网络
    #layer1——conv4_3
    #L2_Normalization网络，来源于ParseNet，源码中是×20，只对Conv4_3进行
    conv4_3_Norm = x = Lambda(L2_Normalization, output_shape=L2_Normalization_Shape, name='l2_normlization')(conv4_3)

    k = 4
    x = Conv2D(filters = num_classes*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation=None, padding='same', name='conv4_3_norm_cls')(conv4_3_Norm)
    #(38,38,kc)
    conv4_3_cls = Reshape((-1,num_classes), name='conv4_3_cls')(x)
    #conv4_3_cls = Flatten(name='conv4_3_cls')(x)     #拉平

    x = Conv2D(filters = 4*k, kernel_size=(3,3), data_format='channels_last', #回归网络
                activation='sigmoid', padding='same', name='conv4_3_norm_reg')(conv4_3_Norm)
    conv4_3_reg = Reshape((-1, 4), name='conv4_3_reg')(x)
    #conv4_3_reg = Flatten(name='conv4_3_reg')(x)             #拉平
    #layer2——conv7
    k = 6
    x = Conv2D(filters = num_classes*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation=None, padding='same', name='conv7_cls_before_flatten')(conv7)
    conv7_cls = Reshape((-1,num_classes), name='conv7_cls')(x)
    #conv7_cls = Flatten(name='conv7_cls')(x)
    x = Conv2D(filters = 4*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation='sigmoid', padding='same', name='conv7_reg_before_flatten')(conv7)
    conv7_reg = Reshape((-1, 4), name='conv7_reg')(x)
    #conv7_reg = Flatten(name='conv7_reg')(x)
    #layer3——conv8_2
    k = 6
    x = Conv2D(filters = num_classes*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation=None, padding='same', name='conv8_2_cls_before_flatten')(conv8_2)
    conv8_2_cls = Reshape((-1,num_classes), name='conv8_2_cls')(x)
    #conv8_2_cls = Flatten(name='conv8_2_cls')(x)
    x = Conv2D(filters = 4*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation='sigmoid', padding='same', name='conv8_2_reg_before_flatten')(conv8_2)
    conv8_2_reg = Reshape((-1, 4), name='conv8_2_reg')(x)
    #conv8_2_reg = Flatten(name='conv8_2_reg')(x)
    #layer4——conv9_2
    k = 6
    x = Conv2D(filters = num_classes*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation=None, padding='same', name='conv9_2_cls_before_flatten')(conv9_2)
    conv9_2_cls = Reshape((-1,num_classes), name='conv9_2_cls')(x)
    #conv9_2_cls = Flatten(name='conv9_2_cls')(x)
    x = Conv2D(filters = 4*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation='sigmoid', padding='same', name='conv9_2_reg_before_flatten')(conv9_2)
    conv9_2_reg = Reshape((-1, 4), name='conv9_2_reg')(x)
    #conv9_2_reg = Flatten(name='conv9_2_reg')(x)
    #layer5——conv10_2
    k = 4
    x = Conv2D(filters = num_classes*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation=None, padding='same', name='conv10_2_cls_before_flatten')(conv10_2)
    conv10_2_cls = Reshape((-1,num_classes), name='conv10_2_cls')(x)
    #conv10_2_cls = Flatten(name='conv10_2_cls')(x)
    x = Conv2D(filters = 4*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation='sigmoid', padding='same', name='conv10_2_reg_before_flatten')(conv10_2)
    conv10_2_reg = Reshape((-1, 4), name='conv10_2_reg')(x)
    #conv10_2_reg = Flatten(name='conv10_2_reg')(x)
    #layer6——conv11_2
    k = 4
    x = Conv2D(filters = num_classes*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation=None, padding='same', name='conv11_2_cls_before_flatten')(conv11_2)
    conv11_2_cls = Reshape((-1,num_classes), name='conv11_2_cls')(x)
    #conv11_2_cls = Flatten(name='conv11_2_cls')(x)
    x = Conv2D(filters = 4*k, kernel_size=(3,3), data_format='channels_last', #分类网络
                activation='sigmoid', padding='same', name='conv11_2_reg_before_flatten')(conv11_2)
    conv11_2_reg = Reshape((-1, 4), name='conv11_2_reg')(x)
    #conv11_2_reg = Flatten(name='conv11_2_reg')(x)

    #开始组合这些预测结果
    cls_all = concatenate([conv4_3_cls, conv7_cls, conv8_2_cls, conv9_2_cls, conv10_2_cls, conv11_2_cls], axis=1, name = 'cls_all')
    reg_all = concatenate([conv4_3_reg, conv7_reg, conv8_2_reg, conv9_2_reg, conv10_2_reg, conv11_2_reg], axis=1, name = 'reg_all')
    pred_all = concatenate([cls_all, reg_all], axis=-1, name='pred_all')


    from ssd_losses import ssd_loss
    from keras.optimizers import SGD

    sgd = SGD(lr=1e-4, decay=1e-4, momentum=0.9, nesterov=True)
    model = Model(inputs=inputs, outputs=pred_all)
    model.compile(optimizer=sgd, loss=ssd_loss)#, metrics=['accuracy'])

    


    return model

model = SSD300()
#model.summary()
#from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

from ssd_loaddata import generate_data_from_file, load_data
import numpy as np


x, y_true = load_data(batch_num =0)
model.fit(x, y_true, epochs=30, batch_size=10, validation_split=0.01)

'''
batch_size = 128
batch_num = 39
model.fit_generator(generate_data_from_file(), steps_per_epoch=batch_num, epochs=10)
'''
'''
x, y_true = load_data(batch_num = 39)
model.fit(x, y_true, epochs=10, batch_size=5, verbose=1, validation_split=0.01)
'''


    



    




     



