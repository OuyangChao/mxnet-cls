# Image Classification

This repository is copied from [Image Classification Examples](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification) of [MXNet](https://github.com/apache/incubator-mxnet). And I made some modifications to these examples, including removing some files that I don't need (eg. R programs) and adding some useful files. And I simplfied the README.md, if you want a more detailed description, you should read the original [README.md](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/README.md).

## Contents

1. [How to prepare datasets](#prepare-datasets)
2. [How to train](#train)
3. [How to score](#score)
4. [How to predict an image](#predict)
5. [How to fine-tune](#fine-tune)

## Prepare Datasets

Assume all images are stored as individual image files such as .png or .jpg, and images belonging to the same class are placed in the same directory. All these class directories are then in the same root `data` directory.
* We first prepare two `.lst` files, which consist of the labels and image paths can be used for generating `rec` files.
```bash
python im2rec.py rec/img data/ --list=True --train-ratio=0.8 --recursive=True
```
* Then we generate the `.rec` files. We resize the images such that the short edge is at least 256px.
```bash
python im2rec.py rec/img_train data/ --resize=256
python im2rec.py rec/img_val data/ --resize=256
```
* Use `python im2rec.py --help` to see more options.

## Train

Python training programs is provided (R training programs is not provided here). Use `train_*.py` to train a network on a particular dataset. For example:

* Train the datasets prepared before. There is a rich set of options, one can list them by passing `--help`.
  ```bash
  python train.py --data-train rec/img_train.rec --data-val rec/img_val.rec
  ```

* When the training process is finished, we generate a training log in `log/train.log` and use `parse_log.py` to parse it into a markdown table.
  ```bash
   python parse_log.py log/train.log
  ```

## Score
We can use `score.py` to score a model on a dataset.
```bash
python score.py --model models/img_cls --epoch 30 --data-val rec/img_val.rec
```

## Predict
We can use `predict.py` to predict an image.
```bash
python predict.py img.jpg --model models/img_cls --epoch 30
```

## Fine-tune
TODO


