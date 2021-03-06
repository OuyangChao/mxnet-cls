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

import os
import argparse
import logging
from common import data, fit
import mxnet as mx

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="training",
          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 1)
    parser.set_defaults(
        # network
        network          = 'resnet',
        num_layers       = 18,
        # data
        data_train       = 'rec/img_train.rec',
        data_val         = 'rec/img_val.rec',
        num_classes      = 2,
        image_shape      = '3,224,224',
        # train
        batch_size       = 32,
        num_epochs       = 10,
        lr               = 0.01,
        lr_step_epochs   = '6,8',
        dtype            = 'float32',
        gpus             = '0',
        model_prefix     = 'models/img_cls',
        log              = 'log/train.log'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # check actual number of train_images
    if os.path.exists(args.data_train.replace('.rec', '.idx')):
        with open(args.data_train.replace('.rec', '.idx'), 'r') as f:
            txt = f.readlines()
        args.num_examples = len(txt)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args.log)
    logger.addHandler(fh)

    # train
    fit.fit(args, sym, data.get_rec_iter)
