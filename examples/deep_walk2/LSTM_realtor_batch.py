#!/usr/bin/env python

import climate
import ConfigParser
import io
import numpy as np
import theanets
import scipy.io
import os
import tempfile
import urllib
import zipfile
import pdb
import glob
import random
import sys

logging = climate.get_logger('lstm-chime')

climate.enable_default_logging()

def map_train_val_id(fn, fn_train_id, fn_val_id):
    dict_train = {}
    dict_val = {}
    with open(fn_train_id, 'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            dict_train[parts[0]] = 1

    with open(fn_val_id, 'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            dict_val[parts[0]] = 1
    idx = 0
    id2train = {}
    id2val = {}
    with open(fn,'r') as fid:
        for aline in fn:
            parts = aline.strip().split()
            if parts[0] in dict_train:
                #dict_train[parts[0]] = idx
                id2train[idx] = 1
            else:
                #dict_val[parts[0]] = idx
                id2val[idx] = 1
            idx += 1
    return id2train, id2val

def load_path_and_split(path_fn, ratio = 0.8):
    path = np.loadtxt(path_fn, dtype='int32')
    split = int(path.shape[0] * ratio)
    train_path = path[0:split,:]
    val_path = path[split:,:]
    return train_path, val_path
def load_path(path_fn):
    path = np.loadtxt(path_fn, dtype='int32')
    return path


def load_price(fn):
    dict_fn2price = {}
    idx = 0
    with open(fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            parts = aline.split()
            dict_fn2price[idx] = float(parts[1])
            idx += 1
    np_price = np.zeros((idx,1), dtype='float')
    for i in range(idx):
        np_price[i,0] = dict_fn2price[i]
    return np_price

def load_fea(fea_fn, fn):
    dict_id = {}
    idx = 0
    with open(fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            parts = aline.split()
            dict_id[parts[0]] = float(parts[1])
            idx += 1
    dict_fea = {}
    fea_num = 0
    with open(fea_fn, 'r') as fid:
        for aline in fid:
            aline = aline.strip()
            parts = aline.split()
            fea = [ float(p) for p in parts[1:] ]
            dict_fea[parts[0]] =  fea
            if fea_num > 0 and fea_num != len(fea):
                print 'Should not happen'
                sys.exit()
            fea_num = len(fea)

    numpy_fea = np.zeros((len(dict_id), fea_num))
    for house in dict_id:
        numpy_fea[dict_id[house], :] = np.asarray(dict_fea[house])
    return numpy_fea 

def main(layer_str, model_dir, fea_fn, train_path_fn, val_path_fn, fn, fn_train_id, fn_val_id, **kwargs):
#def main(layer_nums, data_dir, model_dir, val_fn, **kwargs):

    layer_nums = layer_str.split(',')
    layer_nums = [ int(num) for num in layer_nums ]
    hid_l1 = None
    if 'hid_l1' in kwargs:
        hid_l1 = float(kwargs['hid_l1'])
    l1 = None
    if 'l1' in kwargs:
        l1 = float(kwargs['l1'])
    l2 = None
    if 'l2' in kwargs:
        l2 = float(kwargs['l2'])

    id2train, id2val = map_train_val_id(fn, fn_train_id, fn_val_id)
    train_path = load_path(train_path_fn)
    val_path = load_path(val_path_fn)
    train_idx = range(train_path.shape[0])
    val_idx = range(val_path.shape[0])
    batch_size = 256

    np_fea = load_fea(fea_fn, fn)
    np_price = load_price(fn)
    T = 10
    fea_num = np_fea.shape[1]

    train_batch_x = np.zeros( (T, batch_size, fea_num), dtype=np.float32)
    train_batch_y = np.zeros( (T, batch_size, 1), dtype = np.float32)
    train_mask = np.zeros((T, batch_size, 1), dtype = np.float32)

    val_batch_x = np.zeros( (T, batch_size, fea_num), dtype=np.float32)
    val_batch_y = np.zeros( (T, batch_size, 1), dtype = np.float32)
    val_mask = np.zeros((T, batch_size, 1), dtype = np.float32)

    def batch_train():
        random.shuffle(train_idx)
        for i in range(batch_size):
            train_batch_x[:,i,:] = np_fea[train_path[train_idx[i]],:]
            train_batch_y[:,i,:] = np_price[train_path[train_idx[i]],:]
        train_mask[:] = 1.0
        #for i in range(batch_size):
        #    for j in range(T):
        #        if train_path[train_idx[i],j] in val_mask:
        #            train_mask[j,i,0] = 0
        return [ train_batch_x, train_batch_y , train_mask]
    def batch_val():
        random.shuffle(val_idx)
        for i in range(batch_size):
            val_batch_x[:,i,:] = np_fea[val_path[val_idx[i]],:]
            val_batch_y[:,i,:] = np_price[val_path[val_idx[i]],:]
        val_mask[:] = 1.0
        #for i in range(batch_size):
        #    for j in range(T):
        #        if val_path[val_idx[i],j] in val_mask:
        #            val_mask[j,i,0] = 0
        return [ val_batch_x, val_batch_y, val_mask]
    
    def layer(n):
        return dict(form='bidirectional', worker='lstm', size=n)

    build_layers = [ fea_num ]
    lstm_layers = [ layer(num) for num in layer_nums]
    [ build_layers.append(lstm_layer) for lstm_layer in lstm_layers ]
    build_layers.append(1)
    build_layers = tuple(build_layers)

    
    e = theanets.Experiment(
        theanets.recurrent.Regressor,
        layers=build_layers,
        weighted=True
    )
    
    layer_str = layer_str.replace(',','-')
    #val_base = os.path.splitext(os.path.basename(val_fn))[0]
    save_fn = os.path.join(model_dir, 'models-{0}-{1}-{2}-{3}-{4}.pkl'.format(layer_str, batch_size, hid_l1, l1, l2))
    print save_fn

    e.train(
        batch_train,
        batch_val,
        algorithm='rmsprop',
        learning_rate=0.001,
        gradient_clip=1,
        train_batches=30,
        valid_batches=3,
        batch_size=batch_size,
        weight_l2 = l2,
        weight_l1 = l1,
        hid_l1 = hid_l1
    )
    e.save(save_fn)

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print "Usage: {0} <conf_fn>".format(sys.argv[0])
        sys.exit()
    cf = ConfigParser.ConfigParser()
    cf.read(sys.argv[1])
    fn = cf.get('INPUT', 'fn') # this could be the id as well as the price.
    fn_train_id = cf.get('INPUT', 'fn_train_id')
    fn_val_id = cf.get('INPUT', 'fn_val_id')
    layer_str = cf.get('INPUT', 'layer')
    l1 = cf.get('INPUT', 'l1')
    l2 = cf.get('INPUT', 'l2')
    hid_h1 = cf.get('INPUT', 'hid_l1')
    fea_fn = cf.get('INPUT', 'fea_fn')
    train_path_fn = cf.get('INPUT', 'train_path_fn')
    val_path_fn = cf.get('INPUT', 'val_path_fn')

    model_dir = cf.get('OUTPUT', 'model_dir')
    kwargs = dict(l1 = l1, l2 = l2, hid_h1 = hid_h1)

    main(layer_str, model_dir, fea_fn, train_path_fn, val_path_fn, fn, fn_train_id, fn_val_id, **kwargs)

