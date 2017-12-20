import mxnet as mx
import matplotlib.pyplot as plt
import cv2
import numpy as np
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""

sym, arg_params, aux_params = mx.model.load_checkpoint('models/planeship', 30)
all_layers = sym.get_internals()
fe_sym = all_layers['flatten0_output']
mod = mx.mod.Module(symbol=fe_sym, context=mx.gpu(0), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params)
with open('models/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

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

def predict(url):
    img = get_image(url, show=False)
    # compute the predict probabilities
    mod.forward(Batch([mx.nd.array(img)]))
    features = mod.get_outputs()[0].asnumpy()
    print(features.shape)

predict('data/plane/image01_041.jpg')
