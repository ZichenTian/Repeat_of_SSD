import numpy as np
import cv2
import os
import random
import pickle

data_path = 'datasave/'


def load_data(data_path = data_path, batch_num = 0):
    img_filename = 'img' + str(batch_num) + '.pk'
    label_filename = 'label' + str(batch_num) + '.pk'
    img_filename = os.path.join(data_path, img_filename)
    label_filename = os.path.join(data_path, label_filename)
    with open(img_filename, 'rb') as f1, open(label_filename, 'rb') as f2:
        x = pickle.load(f1)     #x是压缩并转换过RGB的图像的numpy，(pic_num, 300, 300, 3)
        y_true = pickle.load(f2)    #y_true是label信息和reg信息，(pic_num, 25)，其中25:(0-20:oneshot cls label, 21-24:reg(cx,cy,w,h))
    return x, y_true



def generate_data_from_file(data_path = data_path, batch_size = 8):
    max_num = int(len(os.listdir(data_path))/2-1)
    max_train_num = max_num - 1
    while True:
        batch_num = random.randint(0,max_train_num)
        x, y_true = load_data(data_path, batch_num)
        total_batch_size = x.shape[0]
        p = list(range(0,total_batch_size))
        for i in range(0, int(total_batch_size/batch_size)):
            mask = random.sample(p, batch_size)
            x1 = x[mask]
            y_true1 = y_true[mask]
            yield(x1, y_true1)

'''
def generate_data_from_file(data_path = data_path, batch_size = 16):
    max_num = int(len(os.listdir(data_path))/2-1)
    max_train_num = max_num - 1
    while True:
        batch_num = random.randint(0,max_train_num)
        x, y_true = load_data(data_path, batch_num)
        yield(x, y_true)
'''
def get_valid_data(data_path = data_path, batch_size = 8):
    max_num = int(len(os.listdir(data_path))/2-1)
    val_num = max_num
    x, y_true = load_data(data_path, val_num)
    mask = list(range(0, batch_size))
    x = x[mask]
    y_true = y_true[mask]
    return x, y_true




