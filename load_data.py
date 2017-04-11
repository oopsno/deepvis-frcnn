# encoding: UTF-8
from __future__ import print_function

import xml.etree.cElementTree as ETree
import cv2


def load_image(filename):
    """
    load an image into a caffe-compatible numpy.ndarray
    :param filename: filename of the image
    :return: image
    """
    im = cv2.imread(filename)
    im = cv2.resize(im, (227, 227))
    #       BGR => RGB & H x W x C => C x H x W
    im = im[:, :, ::-1].transpose((2, 0, 1))
    # TODO im = im - mean
    return im


def parse_roi_xml(filename):
    """
    Parses a Kaggle flavoured XML definition
    :param filename: filename of the definition
    :return: image_folder, image_filename, bound_boxes
    """
    root = ETree.parse(filename)
    image_folder = root.find('folder').text
    image_filename = root.find('filename/item').text
    bound_boxes = []
    for i, obj in enumerate(root.iter('object')):
        bbox = [i]
        for item in obj.find('bndbox').iter():
            if item.tag != 'bndbox':
                value = int(item.text)
                bbox.append(value)
        bound_boxes.append(bbox)
    return image_folder, image_filename, bound_boxes
