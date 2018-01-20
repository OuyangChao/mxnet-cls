"""
LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
Gradient-based learning applied to document recognition.
Proceedings of the IEEE (1998)
"""
import mxnet as mx

def get_loc(data, attr={'lr_mult':'0.01'}):
    """
    the localisation network in lenet-stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    loc = mx.symbol.Convolution(data=data, num_filter=30, kernel=(5, 5), stride=(2,2))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data=loc, kernel=(2, 2), stride=(2, 2), pool_type='max')
    loc = mx.symbol.Convolution(data=loc, num_filter=60, kernel=(3, 3), stride=(1,1), pad=(1, 1))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data=loc, global_pool=True, kernel=(2, 2), pool_type='avg')
    loc = mx.symbol.Flatten(data=loc)
    loc = mx.symbol.FullyConnected(data=loc, num_hidden=6, name="stn_loc", attr=attr)
    return loc


def get_symbol(num_classes=10, add_stn=False, **kwargs):
    data = mx.symbol.Variable('data')
    if add_stn:
        data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),
                                         transform_type="affine", sampler_type="bilinear")
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(9,9), num_filter=32)
    pool1 = mx.symbol.Pooling(data=conv1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    relu1 = mx.symbol.Activation(data=pool1, act_type="relu")
    # second conv
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(7,7), num_filter=32)
    pool2 = mx.symbol.Pooling(data=conv2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    relu2 = mx.symbol.Activation(data=pool2, act_type="relu")
    # first fullc
    flatten = mx.symbol.Flatten(data=relu2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes)
    # loss
    mnist_cnn = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return mnist_cnn
