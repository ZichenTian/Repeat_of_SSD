try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np

dataset_path = 'dataset/VOCdevkit/VOC2007/'

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
    pic_shape = [int(size_elem.find("width").text), 
                 int(size_elem.find("height").text), 
                 int(size_elem.find("depth").text)]
    annotations = []
    for obj_elem in root.findall("object"):
        anno = {}
        anno['name'] = obj_elem.find("name").text
        anno['truncated'] = (obj_elem.find("truncated").text == '1')
        anno['difficult'] = (obj_elem.find("difficult").text == '1')
        bbox = []
        for coor in obj_elem.find('bndbox'):
            bbox.append(int(coor.text))     #顺序是(xmin, ymin, xmax, ymax)
        anno['bbox'] = bbox
        annotations.append(anno)
        
    return pic_shape, annotations

    
def Test_Read_XML(root_path=dataset_path):
    image_path = os.path.join(root_path, 'JPEGImages')
    anno_path = os.path.join(root_path, 'Annotations')
    for image in os.listdir(image_path):
        anno = os.path.splitext(image)[0] + '.xml'
        image = os.path.join(image_path, image)
        anno = os.path.join(anno_path, anno)
        [shape, annotations] = Read_Annotations(anno)
        srcImage = cv2.imread(image)                            #opencv默认格式是BGR，使用该函数改为RGB
        #srcImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)    #不过使用imshow时还会认为是BGR，导致图像偏蓝
        cv2.imshow('srcImage', srcImage)
        cv2.waitKey()


Test_Read_XML()


    
