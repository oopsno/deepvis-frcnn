#! /usr/bin/env python
# encoding=UTF-8

from __future__ import print_function
from pylab import *

import caffe
caffe.set_mode_gpu()

def load_labels():
    return ['nothing', 'licence']
    
def load_trained_net(model_prototxt = None, model_weights = None):
    assert (model_prototxt is None) == (model_weights is None), 'Specify both model_prototxt and model_weights or neither'
    if model_prototxt is None:
        load_dir = '/home/jyosinsk/results/140311_234854_afadfd3_priv_netbase_upgraded/'
        model_prototxt = load_dir + 'deploy_1.prototxt'
        model_weights = load_dir + 'caffe_imagenet_train_iter_450000'

    print('LOADER: loading net:')
    print('  ', model_prototxt)
    print('  ', model_weights)
    net = caffe.Classifier(model_prototxt, model_weights)
    #net.set_phase_test()

    return net

    
def load_imagenet_mean():
    imagenet_mean = np.load('ilsvrc_2012_mean.npy')
    imagenet_mean = imagenet_mean[:, 14:14+227, 14:14+227]    # (3,256,256) -> (3,227,227) Crop to center 227x227 section
    return imagenet_mean.transpose([2, 0 ,1])
