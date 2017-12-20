#########################################################################
# File Name: run.sh
# Description:
# Author: OuyangChao
# mail: ouyangchao16@163.com
# Created Time: Wed 20 Dec 2017 02:19:22 PM CST
# Last modified: Wed 20 Dec 2017 02:19:22 PM CST
#########################################################################
#!/usr/bin/env sh
set -e

# prepare datasets
echo 'prepare datasets...'
python im2rec.py rec/img data/ --list=True --train-ratio=0.8 --recursive=True
python im2rec.py rec/img_train data/ --resize=256
python im2rec.py rec/img_val data/ --resize=256
echo 'prepare data done.'

# train
python train.py --data-train rec/img_train.rec --data-val rec/img_val.rec

# parse log
python parse_log.py log/train.log

# test
python score.py --model models/img_cls --epoch 30 --data-val rec/img_val.rec

# predict an image
python predict.py img.jpg --model models/img_cls --epoch 30

