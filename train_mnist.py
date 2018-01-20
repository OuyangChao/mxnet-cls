# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import struct

def read_data(label, image, base_url):
    """
    download and read data into numpy
    """
    with open(os.path.join(base_url,label), 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with open(os.path.join(base_url,image), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte', 'train-images-idx3-ubyte', args.data_root)
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte', args.data_root)
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--data-root', type=str, default='data/mnist',
                        help='the root directory of mnist')

    parser.add_argument('--add_stn',  action="store_true", default=False, help='Add Spatial Transformer Network Layer (lenet only)')

    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'lenet',
        # train
        gpus           = '0',
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 30,
        lr             = .05,
        lr_step_epochs = '20',
        log            = 'log/train.log'
    )
    args = parser.parse_args()

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.log)
    logger.addHandler(fh)

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)
