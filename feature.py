import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""


# data: (n, height, width) OR (n, height, width, 3)
def vis_square(data):
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    plt.axis('off')
    plt.show()


def get_image(url, show=False):
    # download and show the image
    # fname = mx.test_utils.download(url)
    fname = url
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
         plt.show()
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    rgb_mean = np.array([123.68,116.779,103.939]).reshape((1,3,1,1))
    return img - rgb_mean


def get_mnist(url, show=False):
    # download and show the image
    # fname = mx.test_utils.download(url)
    fname = url
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)
    if img is None:
         return None
    if show:
         plt.imshow(img)
         plt.axis('off')
         plt.show()
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 1, 28, 28).astype(np.float32)/255
    return img


def main():
    if (args.dataset == 'Imagenet'):
        img = get_image(args.image, show=False)
    elif (args.dataset == 'Mnist'):
        img = get_mnist(args.image, show=False)
    else:
        print('Error in dataset')
        return
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model, args.epoch)
    all_layers = sym.get_internals()
    fe_sym = all_layers[args.feat_layer + '_output']
    mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(0), label_names=None)
    if (args.dataset == 'Imagenet'):
        mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
                 label_shapes=mod._label_shapes)
    elif (args.dataset == 'Mnist'):
        mod.bind(for_training=False, data_shapes=[('data', (1,1,28,28))], 
                 label_shapes=mod._label_shapes)
    else:
        print('Error in dataset')
        return
    mod.set_params(arg_params, aux_params)
    mod.forward(Batch([mx.nd.array(img)]))
    features = mod.get_outputs()[0].asnumpy()
    print(features.shape)
    vis_square(features[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize feature maps')
    parser.add_argument('image', type=str, help='Image to be processed')
    parser.add_argument('feat_layer', type=str, help='Layer name to be processed')
    parser.add_argument('--model', type=str, required=True,
                        help = 'the model name.')
    parser.add_argument('--epoch', type=int, default=0,
                        help='load the model on an epoch')
    parser.add_argument('--dataset', type=str, choices=['Mnist', 'Imagenet'], default='Imagenet',
                        help='Imagenet or Mnist')
    args = parser.parse_args()
    main()

