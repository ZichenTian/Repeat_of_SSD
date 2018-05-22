# Repeat of SSD——SSD源码复现（Keras）
## 原论文
SSD:Single Shot MultiBox Detector

经典的目标检测算法，融合了YOLO v1的单阶段检测思想，使用了多尺度检测，借用了Faster RCNN的anchor的概念。

复现这篇文章，主要是因为行人检测领域的F-DNN在Caltech数据集上，小尺度效果远远领先于其他算法。其原因很有可能是因为SSD在小尺度目标上有较好的性能，提供了更多的proposals（F-DNN的前置网络使用的是fine-tune过的SSD）。因此这里先使用Keras复现一遍SSD。

## 系统环境
* opensuse leap 42.2
* tensorflow-gpu 1.8.0
* python 3.4.6
* cuda 9.0
* cudnn 7.1
* keras 2.1.6
* opencv 3.4.0(python)

## 文件架构
* ssd.py：网络构架与编译
* ssd_voc_preprocessing.py：VOC2007数据集预处理，包括XML的解析，anchor的预处理等
* ssd_utils.py：一些工具与子函数

## 使用方法
还没写完。。。差了很多呢。。。

