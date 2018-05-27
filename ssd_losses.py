import keras.backend as K
import tensorflow as tf

def smooth_L1(x):
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def ssd_loss(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    anchors_num = tf.shape(y_pred)[1]

    y_true_all = tf.reshape(y_true, [-1,25])
    y_pred_all = tf.reshape(y_pred, [-1,25])

    y_true_cls = tf.slice(y_true_all, [0,0], [batch_size*anchors_num,21])
    y_true_reg = tf.slice(y_true_all, [0,21], [batch_size*anchors_num,4])
    y_pred_cls = tf.slice(y_pred_all, [0,0], [batch_size*anchors_num,21])
    y_pred_reg = tf.slice(y_pred_all, [0,21], [batch_size*anchors_num,4])

    fpmask = tf.reduce_sum(tf.slice(y_true_cls, [0,1], [batch_size*anchors_num,20]), axis=1)
    fnmask = 1 - fpmask

    p_num = tf.reduce_sum(fpmask)
    n_num = p_num * 3

    #conf_loss_all = tf.nn.softmax_cross_entropy_with_logits(y_pred_cls, y_true_cls)
    conf_loss_all = tf.nn.softmax_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
    conf_loss_n = conf_loss_all * fnmask

    val, index = tf.nn.top_k(conf_loss_n, k=tf.cast(n_num, tf.int32))
    top_k_threshhold = val[-1]  #val会默认从大到小排序，因此取出最小的值来
    hnmask = tf.cast(conf_loss_n >= top_k_threshhold, tf.float32)

    p_conf_loss = tf.reduce_sum(conf_loss_all * fpmask)
    hn_conf_loss = tf.reduce_sum(conf_loss_all * hnmask)

    smooth_L1_loss_all = smooth_L1(y_true_reg - y_pred_reg)
    reg_loss1 = tf.reduce_sum(smooth_L1_loss_all, axis=1) * fpmask
    reg_loss = tf.reduce_sum(reg_loss1)
    '''
    cls_loss = p_conf_loss + hn_conf_loss
    loss_all = cls_loss + reg_loss
    loss = tf.reduce_sum(loss_all / p_num)
    '''

    #loss = tf.reduce_sum(reg_loss / p_num)
    loss = (p_conf_loss + reg_loss + hn_conf_loss) / p_num
    return loss


    


    
    
    
    




