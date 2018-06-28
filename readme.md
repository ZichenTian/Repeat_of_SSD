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
* ssd.py：网络构架与编译，网络训练及预测
* ssd_voc_preprocessing.py：VOC2007数据集预处理，包括XML的解析，anchor的预处理等
* ssd_utils.py：一些工具与子函数
* ssd_losses.py：使用tensorflow计算loss
* ssd_getoutput.py：读取网络预测的结果（pickle文件），并框出物体
* ssd_loaddata.py：读取经ssd_voc_preprocessing.py处理出来的pickle，用生成器的方式喂给网络

## 使用方法
* 下载VOC2007 Trainval数据集，解压放置在dataset下
* 根目录下 `python ssd_voc_preprocessing.py`，会在datasave中生成一系列pickle文件
* 下载VGG16的模型权重文件，notop的即可，放在pretrain下
* 使用 `python ssd.py`进行训练
* 使用 `python ssd_getoutput.py`查看检测效果

