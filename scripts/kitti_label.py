import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('kitti', 'train'), ('kitti', 'val')]

classes = ["car"]

FractionTrain = 0.8
TrainSplit = str(int(FractionTrain*100)) + '-' + str(int((100-FractionTrain*100)))

darknet_format_IS_dir = '/home/amitsinha/data/PASCAL-VOC-CUSTOM-KITTI/VOC2012-CUSTOM-KITTI/darknet-format-ImageSets/'
darknet_format_labels_dir = '/home/amitsinha/data/PASCAL-VOC-CUSTOM-KITTI/VOC2012-CUSTOM-KITTI/labels/'
kitti_pascal_dir = '/home/amitsinha/data/PASCAL-VOC-CUSTOM-KITTI/VOC2012-CUSTOM-KITTI/'

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open(os.path.join(kitti_pascal_dir, 'Annotations/%s.xml'%(image_id)))
    out_file = open(os.path.join(darknet_format_labels_dir, '%s.txt'%(image_id)), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text.lower().strip()
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for db, image_set in sets:
    if not os.path.exists(darknet_format_IS_dir):
        os.makedirs(darknet_format_IS_dir)
    if not os.path.exists(darknet_format_labels_dir):
        os.makedirs(darknet_format_labels_dir)

    image_ids = open(os.path.join(kitti_pascal_dir, 'ImageSets', 'Main/%s/%s.txt'%(TrainSplit, image_set))).read().strip().split()

    list_file = open(os.path.join(darknet_format_IS_dir, '%s.txt'%(image_set)), 'w')
    for image_id in image_ids:
        list_file.write(os.path.join(kitti_pascal_dir, 'JPEGImages/%s.png\n'%(image_id)))
        convert_annotation(image_id)
    list_file.close()

