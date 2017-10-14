import sys
sys.path.insert(0, '../python-torchfile')
import torchfile
from keras.layers import Input, Conv2D, Layer, MaxPool2D, UpSampling2D, Activation, ZeroPadding2D
from keras.layers.merge import Add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.activations import relu, sigmoid
from keras import backend as K
import numpy as np
import tensorflow as tf

def getr(tobj, ln):
    for module in tobj['modules']:
        if ln == 0:
            return module, 0
        ln -= 1
        if 'modules' in module.__dir__():
            got, ln = getr(module, ln)
            if got is not None:
                return got, 0

    return None, ln

def get(tobj, ln):
    tout, _ = getr(tobj, ln)
    return tout

def reorder(tensor4):
    tensor4 = np.swapaxes(tensor4, 0, 2)
    tensor4 = np.swapaxes(tensor4, 1, 3)
    return tensor4

class Conv(Conv2D):
    def __init__(self, output_dim, size, strides=1, padding='valid', tobj=None):
        self.output_dim = output_dim
        self.size = size
        self.tobj = tobj

        super(Conv, self).__init__(output_dim,
                                   size,
                                   strides=strides,
                                   padding=padding,
                                   data_format='channels_first')

    def build(self, input_shape):
        super(Conv, self).build(input_shape)
        weight = reorder(self.tobj['weight'])
        self.set_weights((weight, self.tobj['bias']))

    def call(self, inputs):
        out = inputs
        if self.size == 3:
            out = ZeroPadding2D(padding=1, data_format='channels_first')(inputs)
        return super(Conv, self).call(out)

    def compute_output_shape(self, input_shape):
        return super(Conv, self).compute_output_shape(input_shape)

class BatchNorm(BatchNormalization):
    def __init__(self, tobj=None):
        self.tobj = tobj
        super(BatchNorm, self).__init__(axis=1,
                                        momentum=.1,
                                        epsilon=1e-5)
    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        super(BatchNorm, self).set_weights((self.tobj['weight'],
                         self.tobj['bias'],
                         self.tobj['running_mean'],
                         self.tobj['running_var']))

    def call(self, inputs):
        return super(BatchNorm, self).call(inputs)

    def compute_output_shape(self, input_shape):
        return super(BatchNorm, self).compute_output_shape(input_shape)

class Conv131(Layer):
    def __init__(self, input_dim, tobj=None, first_layer=False):
        self.input_dim = input_dim
        self.tobj = tobj
        self.first_layer = first_layer
        super(Conv131, self).__init__()

    def build(self, input_shape):
        super(Conv131, self).build(input_shape)

    def call(self, inputs, **kwargs):
        tbn1 = get(self.tobj, 0)
        tbn2 = get(self.tobj, 3)
        tbn3 = get(self.tobj, 6)

        tconv1 = get(self.tobj, 2)
        tconv2 = get(self.tobj, 5)
        tconv3 = get(self.tobj, 8)

        batch_norm1 = BatchNorm(tobj=tbn1)(inputs)
        relu1 = relu(batch_norm1)

        output_dim1 = self.input_dim if self.first_layer else self.input_dim / 2
        conv1 = Conv(output_dim1, 1, tobj=tconv1)(relu1)

        batch_norm2 = BatchNorm(tobj=tbn2)(conv1)
        relu2 = relu(batch_norm2)
        conv2 = Conv(output_dim1, 3, tobj=tconv2)(relu2)

        batch_norm3 = BatchNorm(tobj=tbn3)(conv2)
        relu3 = relu(batch_norm3)
        conv3 = Conv(output_dim1 * 2, 1, tobj=tconv3)(relu3)

        return conv3

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[1] = input_shape[1] * (2 if self.first_layer else 1)
        return tuple(input_shape)

class UpSamplingBilinear(Layer):
    def __init__(self, scale=4):
        self.scale = scale
        super(UpSamplingBilinear, self).__init__()

    def build(self, input_shape):
        self.shape = input_shape
        super(UpSamplingBilinear, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_size = self.compute_output_shape(self.shape)[-2:]
        x = K.permute_dimensions(inputs, [0, 2, 3, 1])
        x = tf.image.resize_images(x, new_size)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        return x

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[2] *= self.scale
        input_shape[3] *= self.scale
        return tuple(input_shape)

def add_conv(input_dim, tobj, l):
    return Add()([
        Conv131(input_dim, tobj=get(tobj, 1), first_layer=True)(l),
        Conv(input_dim * 2, 1, tobj=get(tobj, 12))(l)
    ])

def add_id(input_dim, tobj, l):
    return Add()([
        Conv131(input_dim, tobj=get(tobj, 1), first_layer=False)(l),
        l
    ])

pi = 0

def part(tobj, input):
    global pi
    print("Part {}".format(pi))
    pi += 1

    a, b = tobj['modules'][:3:2]

    top_seq, bottom_seq = a['modules']

    top_seq_modules = top_seq['modules']
    top = input

    for module in top_seq_modules[:3]:
        top = add_id(256, module, top)

    if len(top_seq_modules) > 3:
        top = part(top_seq_modules[3], top)

    bottom_seq_modules = bottom_seq['modules']

    bottom = MaxPool2D(data_format='channels_first')(input)

    for module in bottom_seq_modules[1:-1]:
        bottom = add_id(256, module, bottom)

    bottom = UpSampling2D(data_format='channels_first')(bottom)

    out = Add()([top, bottom])

    return add_id(256, b, out)

def half(tobj, input):
    top = tobj['modules'][1]

    seq = part(top, input)

    c, d = top['modules'][-2:]

    for s in (c, d):
        seq = Conv(256, 1, tobj=get(s,0))(seq)
        seq = BatchNorm(tobj=get(s, 1))(seq)
        seq = Activation('relu')(seq)

    return Add()([input, seq])

def model(t):
    global pi
    input = Input((3, 192, 192))

    l = ZeroPadding2D(padding=3, data_format='channels_first')(input)

    l = Conv(64, 7, strides=2, padding='valid', tobj=get(t, 0))(l)

    l = BatchNorm(tobj=get(t, 1))(l)

    l = Activation('relu')(l)

    l = add_conv(64, get(t, 3), l)

    l = MaxPool2D(data_format='channels_first')(l)

    l = add_id(128, get(t, 19), l)

    l = add_conv(128, get(t, 33), l)

    pi = 0
    print("Half 1")
    l = half(get(t, 48), l)

    pi = 0
    print("Half 2")
    l = half(get(t, 689), l)

    l = Conv(256, 1, tobj=get(t, 1333))(l)

    l = BatchNorm(tobj=get(t, 1334))(l)

    l = Activation('relu')(l)

    l = Conv(200, 1, tobj=get(t, 1336))(l)

    l = UpSamplingBilinear()(l)

    l = Activation('sigmoid')(l)

    model = Model(input, l)

    model.compile(loss='mean_squared_error', optimizer='sgd')

    return model

if __name__ == "__main__":
    t = torchfile.load('vrn-unguided.t7')
    model = model(t)
    z = np.zeros((3, 96, 96))
    print(model.predict(z))