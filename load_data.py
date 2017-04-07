# encoding: UTF-8
from __future__ import print_function

import xml.etree.cElementTree as ETree
import cv2


def load_image(filename):
    im = cv2.imread(filename)
    im = cv2.resize(im, (227, 227))
    #       BGR => RGB & H x W x C => C x H x W
    im = im[:, :, ::-1].transpose((2, 0, 1))
    # TODO im = im - mean
    return im


def load_data(filename):
    root = ETree.parse(filename)
    image_filename = root.find('filename/item').text
    bound_boxes = []
    for obj in root.iter('object'):
        bbox = []
        for item in obj.find('bndbox').iter():
            if item.tag != 'bndbox':
                value = int(item.text)
                bbox.append(value)
        bound_boxes.append(bbox)
    return load_image(image_filename), bound_boxes
