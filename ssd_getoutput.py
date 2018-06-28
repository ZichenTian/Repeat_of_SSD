import numpy as np
import cv2
import math

def Anchor_Init():
    '''
    生成论文中的8732个anchor的位置(xmin, ymin, xmax, ymax)
    '''
    fmap_size = (
        (38, 38),(19, 19), (10, 10),
        (5, 5), (3, 3), (1, 1)
    )
    anchor_step = (
        8, 16, 32, 64, 100, 300
    )
    anchor_size = (
        (30., 60.), (60., 111.), (111., 162.),    #跟论文里不能对应起来，是作者caffe下源代码的值
        (162., 213.), (213., 264.), (264., 315.)
    )
    anchor_ratios = (
        (2, 0.5), (2, 0.5, 3, 1./3), (2, 0.5, 3, 1./3), 
        (2, 0.5, 3, 1./3), (2, 0.5), (2, 0.5)
    )

    anchors_loc = []
    for i in range(len(fmap_size)):
        h = []
        w = []
        w.append(anchor_size[i][0])    #min_size
        h.append(anchor_size[i][0])
        w.append(math.sqrt(anchor_size[i][0]*anchor_size[i][1]))  #sqrt(min_size*max_size)
        h.append(math.sqrt(anchor_size[i][0]*anchor_size[i][1]))
        for r in anchor_ratios[i]:
            h.append(anchor_size[i][0]/math.sqrt(r))
            w.append(anchor_size[i][0]*math.sqrt(r))
        #xc, yc = np.mgrid[0:300, 0:300]
        xc = yc = list(range(0, 300, anchor_step[i]))
        for y in yc:
            y += anchor_step[i]/2
            for x in xc:
                x += anchor_step[i]/2
                for j in range(len(w)):
                    anchor = []
                    anchor.append(int(x-w[j]/2))
                    anchor.append(int(y-h[j]/2))
                    anchor.append(int(x+w[j]/2))
                    anchor.append(int(y+h[j]/2))
                    anchors_loc.append(anchor)
    return anchors_loc

def calc_reg(loc, reg):
    dt_w = loc[2]-loc[0]
    dt_h = loc[3]-loc[1]
    dt_cx = (loc[0]+loc[2])/2
    dt_cy = (loc[1]+loc[3])/2

    gt_cx = reg[0]*1*dt_w+dt_cx
    gt_cy = reg[1]*1*dt_h+dt_cy
    gt_w = math.exp(reg[2]*2)*dt_w
    gt_h = math.exp(reg[3]*2)*dt_h

    gt_xmin = int(gt_cx-gt_w/2)
    gt_xmax = int(gt_cx+gt_w/2)
    gt_ymin = int(gt_cy-gt_h/2)
    gt_ymax = int(gt_cy+gt_h/2)

    return [gt_xmin, gt_ymin, gt_xmax, gt_ymax]

def Calc_IoU(bbox1, bbox2):
    '''
    计算IoU
    Inputs：bbox1、bbox2——(xmin, ymin, xmax, ymax)
    Outputs:IoU——[0,1]
    '''
    x_right = max(bbox1[0], bbox1[2], bbox2[0], bbox2[2])
    x_left = min(bbox1[0], bbox1[2], bbox2[0], bbox2[2])
    y_down = max(bbox1[1], bbox1[3], bbox2[1], bbox2[3])
    y_up = min(bbox1[1], bbox1[3], bbox2[1], bbox2[3])

    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    Iw = w1 + w2 - (x_right - x_left)
    Ih = h1 + h2 - (y_down - y_up)

    if(Iw<=0 or Ih<=0):  #不交叠
        return 0
    Ia = Iw*Ih      #交叠面积
    Oa = w1*h1+w2*h2-Ia     #并面积
    IoU = float(Ia)/Oa
    return IoU

def nms(bbox, cls_num, scores, threshold=0.3):
    result = []
    for n_cls in range(0,21):
        mask = []
        for i in range(len(cls_num)):
            if cls_num[i] == n_cls:
                mask.append(i)
        if len(mask) == 0:
            continue
        bbx = np.concatenate((np.array(bbox)[mask], np.array(cls_num)[mask].reshape((-1,1)),
                              np.array(scores)[mask].reshape((-1,1))), axis=1)
        bbx = bbx[np.argsort(-bbx[:,5])]
        while bbx.shape[0] > 0:
            max_b = bbx[0]
            bbx = np.delete(bbx, 0, axis=0)
            result.append(list(max_b))
            if bbx.shape[0] <= 0:
                break
            k = 0
            for b in bbx:
                IoU = Calc_IoU(max_b, b)
                if IoU > threshold:
                    bbx = np.delete(bbx, k, axis=0)
                    continue
                k += 1
    
    return result

label2name = [
    'none',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]

def draw_rectangle(x, bbx, cls_num, scores, color):
    cv2.rectangle(x, (int(bbx[0]), int(bbx[1])), (int(bbx[2]), int(bbx[3])), color, 1)
    cv2.putText(x, label2name[cls_num], (int(bbx[0]),int(bbx[1])), cv2.FONT_HERSHEY_PLAIN, 1, color)
    cv2.imshow('srcImage', x)
    cv2.waitKey()

def anchor_decode(x, y_pred):
    batch_size = x.shape[0]
    anchors_loc = Anchor_Init()
    for i in range(batch_size):
        x1 = cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR)
        y_pred1 = y_pred[i]
        cv2.imshow('srcImage', x1)
        cv2.waitKey()
        y_cls = y_pred1[:,range(0,21)]
        y_reg = y_pred1[:,range(21,25)]
        y_cls_r = np.argmax(y_cls, axis=1)
        print(y_cls_r.shape)
        loc = []
        bbox = []
        scores = []
        cls_num = []
        for j in range(y_cls_r.shape[0]):
            if y_cls_r[j] > 0:
                loc1 = anchors_loc[j]
                bbox1 = calc_reg(loc1, y_reg[j])
                scores1 = y_cls[j, y_cls_r[j]]
                cls_num1 = y_cls_r[j]
                loc.append(loc1)
                bbox.append(bbox1)
                scores.append(scores1)
                cls_num.append(cls_num1)
        r_bbox = nms(loc, cls_num, scores)
        for r_bbx in r_bbox:
            draw_rectangle(x1, r_bbx[0:4], int(r_bbx[4]), r_bbx[5], (0,0,255))
            

    while True:
        pass


'''
def anchor_decode(x, y_pred):
    batch_size = x.shape[0]
    anchors_loc = Anchor_Init()
    for i in range(batch_size):
        x1 = cv2.cvtColor(x[i], cv2.COLOR_RGB2BGR)
        y_pred1 = y_pred[i]
        cv2.imshow('srcImage', x1)
        cv2.waitKey()
        y_cls = y_pred1[:,range(0,21)]
        y_reg = y_pred1[:,range(21,25)]
        y_cls_r = np.argmax(y_cls, axis=1)
        print(y_cls_r.shape)
        for j in range(y_cls_r.shape[0]):
            if y_cls_r[j] > 0:
                loc = anchors_loc[j]
                bbox = calc_reg(loc, y_reg[j])
                cv2.rectangle(x1, (loc[0], loc[1]), (loc[2], loc[3]), (0,0,255), 1)
                cv2.rectangle(x1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 1)
                cv2.imshow('srcImage', x1)
                print(y_cls_r[j])
                cv2.waitKey()
    while True:
        pass
'''

def sp_test(x, y_pred, j, y_cls):
    anchors_loc = Anchor_Init()
    a = y_pred[:,y_cls]
    print(max(a))
    b = np.argsort(a)
    b = b[-3:-1]
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    y_reg = y_pred[:,range(21,25)]
    for j in b:
        loc = anchors_loc[j]
        bbox = calc_reg(loc, y_reg[j])
        cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 1)
        cv2.imshow('srcImage', x)
        cv2.waitKey()


import pickle
img_filename = './test3.pkl'
with open(img_filename, 'rb') as f1:
    y_pred = pickle.load(f1)     #x是压缩并转换过RGB的图像的numpy，(pic_num, 300, 300, 3)
from ssd_loaddata import load_data
x, y_true = load_data(batch_num =39)
#x= x[range(0,16)]

#sp_test(x[1], y_pred[1])

anchor_decode(x, y_pred)



