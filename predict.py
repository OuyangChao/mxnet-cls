import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

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
    # it depends on whether the RGB mean is specified in the training phase.
    rgb_mean = np.array([123.68,116.779,103.939]).reshape((1,3,1,1))
    img = img - rgb_mean
    return img

def predict(model, epoch, url):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model, epoch)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
        label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    img = get_image(url, show=False)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    # print the top-5
    prob = np.squeeze(prob)
    a = np.argsort(prob)[::-1]
    for i in a[0:5]:
        print('probability=%f, class=%s' %(prob[i], labels[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predicting an image')
    parser.add_argument('url', nargs=1, type=str,
                        help = 'the image file for predicting')
    parser.add_argument('--model', type=str, required=True,
                        help = 'the model name.')
    parser.add_argument('--epoch', type=int, default=0,
                        help='load the model on an epoch')
    args = parser.parse_args()
    predict(args.model, args.epoch, args.url)
