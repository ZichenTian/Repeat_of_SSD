try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import math
import pickle

dataset_path = 'dataset/VOCdevkit/VOC2007/'
datasave_path = 'datasave/'

def Read_Annotations(XML_file):
    '''
    读取一个Annotations文件的内容
    Annotations文件是XML格式
    XML的主要内容有：
    1、 图片的尺寸信息（Width,Height,Depth）
    2、 物体标注（name,pose,truncated,difficult,bndbox(xmin,ymin,xmax,ymax)）
        其中truncated代表是否不完整,difficult代表需要借助上下文才能正确识别，在test的时候应当被忽略
    
    inputs: XML_file, XML文件路径
    outputs: pic_shape，图片的尺寸(W,H,D), list
             annotations，标注信息的list，每一个元素为一个dict{'name', 'truncated', 'difficult', 'bbox'}
                          其中bbox为一个list[xmin, ymin, xmax, ymax]
    '''
    tree = ET.parse(XML_file)
    root = tree.getroot()
    size_elem = root.find("size")
    pic_shape = (int(size_elem.find("width").text), 
                 int(size_elem.find("height").text), 
                 int(size_elem.find("depth").text))
    annotations = []
    for obj_elem in root.findall("object"):
        anno = {}
        anno['name'] = obj_elem.find("name").text
        anno['truncated'] = (obj_elem.find("truncated").text == '1')
        anno['difficult'] = (obj_elem.find("difficult").text == '1')
        if(anno['difficult'] == True):      #difficult不去掉
            continue
        #if(anno['truncated'] == True):     #truncated不去掉
        #    continue
        
        bbox = []
        for coor in obj_elem.find('bndbox'):
            bbox.append(int(coor.text))     #顺序是(xmin, ymin, xmax, ymax)
        anno['bbox'] = bbox
        annotations.append(anno)
        
    return pic_shape, annotations

#annotations是list，里面的元素是dict，都是可变对象，因此传的是引用
def Annotations_Resize(shape, size, annotations):       
    '''
    将标签也跟随图像resize的函数
    input：shape——原图像尺寸
           size——新图像尺寸
           annotations——读取出来的标注
    '''
    x_ratio = size[0]/shape[0]
    y_ratio = size[1]/shape[1]
    for anno in annotations:
        anno['bbox'][0] = int(anno['bbox'][0]*x_ratio)
        anno['bbox'][2] = int(anno['bbox'][2]*x_ratio)
        anno['bbox'][1] = int(anno['bbox'][1]*y_ratio)
        anno['bbox'][3] = int(anno['bbox'][3]*y_ratio)

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
        (30., 60.), (60., 111.), (111., 162.),    #跟论文里不能对应起来
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
    
    

def Calc_Reg(gt_bbox, dt_bbox):
    '''
    计算论文中的(2)式——位置回归式
    Inputs：gt和dt的bbox(xmin, ymin, xmax, ymax)
    Outputs: reg(cx, cy, w, h)，是卷积神经网络需要回归的东西
    '''
    gt_cx = (gt_bbox[0]+gt_bbox[2])/2
    gt_cy = (gt_bbox[1]+gt_bbox[3])/2
    gt_w = gt_bbox[2]-gt_bbox[0]
    gt_h = gt_bbox[3]-gt_bbox[1]

    dt_cx = (dt_bbox[0]+dt_bbox[2])/2
    dt_cy = (dt_bbox[1]+dt_bbox[3])/2
    dt_w = dt_bbox[2]-dt_bbox[0]
    dt_h = dt_bbox[3]-dt_bbox[1]

    reg = np.zeros(4)
    reg[0] = float(gt_cx-dt_cx)/dt_w
    reg[1] = float(gt_cy-dt_cy)/dt_h
    reg[2] = math.log(float(gt_w)/dt_w)
    reg[3] = math.log(float(gt_h)/dt_h)

    return reg

name2label = {
    'none': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20,
}
    


def Anchor_Encode(anchors_loc = None, 
                  annotations = None, 
                  threshold = 0.5):
    '''
    根据标注，定义anchors的正负样本，并存储IoU、标签、bbox回归值等
    '''
    scores = np.zeros(len(anchors_loc)) #scores就是IoU
    labels = np.zeros(len(anchors_loc))
    regs = np.zeros((len(anchors_loc), 4))
    for anno in annotations:
        gt_bbox = anno['bbox']  #gt的bbox(xmin, ymin, xmax, ymax)
        for i, anchor in enumerate(anchors_loc):
            IoU = Calc_IoU(anchor, gt_bbox)
            # TODO:暂时没有按照论文原作者的思路来，而是大于0.5的都认为是match了，这样有可能导致小的gt match不到
            if(IoU >= threshold and IoU > scores[i]):   #超过阈值，且比原来match的IoU还大
                labels[i] = name2label[anno['name']]
                scores[i] = IoU
                regs[i] = Calc_Reg(gt_bbox, anchor)
    return scores, labels, regs


def toOneShot(labels):
    num_classes = 21
    oneshot = np.zeros((len(labels), num_classes))
    mask = list(range(0, len(labels)))
    oneshot[mask, labels.astype('int8')] = 1
    return oneshot


def SaveData(x, y_true, batch_num, save_path = datasave_path):
    img_filename = 'img' + str(batch_num) + '.pk'
    label_filename = 'label' + str(batch_num) + '.pk'
    img_filename = os.path.join(save_path, img_filename)
    label_filename = os.path.join(datasave_path, label_filename)
    with open(img_filename, 'wb') as f1, open(label_filename, 'wb') as f2:
        pickle.dump(np.array(x), f1)
        pickle.dump(np.array(y_true), f2)
    



    

def Read_Dataset(root_path=dataset_path, batch_size=128):
    image_path = os.path.join(root_path, 'JPEGImages')
    anno_path = os.path.join(root_path, 'Annotations')
    anchors_loc = Anchor_Init()     #生成anchors的坐标
    x = []
    y_true = []
    cnt = 0
    batch_num = 0
    for image_file in os.listdir(image_path):   #这里list到的只是一个文件名
        print(image_file)
        anno_file = os.path.splitext(image_file)[0] + '.xml'
        image_file = os.path.join(image_path, image_file)
        anno_file = os.path.join(anno_path, anno_file)
        [shape, annotations] = Read_Annotations(anno_file)          #读取XML信息，shape为尺度，annotations为标注信息
        srcImage = cv2.imread(image_file)                            #opencv默认格式是BGR，使用该函数改为RGB
        srcImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)    #不过使用imshow时还会认为是BGR，导致图像偏蓝
        #cv2.imshow('srcImage', srcImage)
        #cv2.waitKey()
        size = (300, 300)
        srcImage = cv2.resize(srcImage, size, interpolation=cv2.INTER_AREA) #定义个x，省得srcImage被改来改去影响数据
        x.append(srcImage)
        #cv2.imshow('srcImage', srcImage)
        #cv2.waitKey()
        Annotations_Resize(shape, size, annotations)    #所有bbox进行resize
        '''
        for annos in annotations:
            bbox = annos['bbox']
            cv2.rectangle(srcImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 1)
        cv2.imshow('srcImage', srcImage)
        cv2.waitKey()
        '''
        scores, labels, regs = Anchor_Encode(anchors_loc, annotations, 0.5) #计算正负样本及BBR回归值
        '''
        for i in range(len(anchors_loc)):
            if(labels[i] > 0):
                print(labels[i], scores[i], regs[i])
                bbox = anchors_loc[i]
                cv2.rectangle(srcImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 1)
                cv2.imshow('srcImage', srcImage)
                cv2.waitKey()
        '''
        labels = toOneShot(labels)
        labels = np.concatenate((labels, regs), axis=1)
        y_true.append(labels)   #y_true的格式：21个one-shot用于标记类别，4个用于表示reg(cx, cy, w, h)
        cnt += 1
        
        if(cnt % batch_size == 0):
            SaveData(x, y_true, batch_num)
            print('batch_num= % d', batch_num)
            x = []
            y_true = []
            batch_num += 1
    SaveData(x, y_true, batch_num)
    print('batch_num= % d', batch_num)


        
        
            



        
        


Read_Dataset(dataset_path, 128)


    
