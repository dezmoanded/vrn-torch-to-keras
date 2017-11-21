import sys
sys.path.insert(0, '../python-torchfile')
import torchfile
from keras.layers import Input, MaxPool2D, UpSampling2D, Activation, ZeroPadding2D
from keras.layers.merge import Add
from keras.models import Model
import numpy as np

from custom_layers import Conv, BatchNorm, UpSamplingBilinear

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

def conv131(input_dim, inputs, tobj=None, first_layer=False):
    tbn1 = get(tobj, 0)
    tbn2 = get(tobj, 3)
    tbn3 = get(tobj, 6)

    tconv1 = get(tobj, 2)
    tconv2 = get(tobj, 5)
    tconv3 = get(tobj, 8)

    batch_norm1 = BatchNorm(tobj=tbn1)(inputs)
    relu1 = Activation('relu')(batch_norm1)

    output_dim1 = input_dim if first_layer else input_dim / 2
    conv1 = Conv(output_dim1, 1, tobj=tconv1)(relu1)

    batch_norm2 = BatchNorm(tobj=tbn2)(conv1)
    relu2 = Activation('relu')(batch_norm2)
    conv2 = Conv(output_dim1, 3, tobj=tconv2)(relu2)

    batch_norm3 = BatchNorm(tobj=tbn3)(conv2)
    relu3 = Activation('relu')(batch_norm3)
    conv3 = Conv(output_dim1 * 2, 1, tobj=tconv3)(relu3)

    return conv3

def add_conv(input_dim, tobj, l):
    return Add()([
        conv131(input_dim, l,tobj=get(tobj, 1), first_layer=True),
        Conv(input_dim * 2, 1, tobj=get(tobj, 12))(l)
    ])

def add_id(input_dim, tobj, l):
    return Add()([
        conv131(input_dim, l, tobj=get(tobj, 1), first_layer=False),
        l
    ])

pi = 0

def part(tobj, input, i):
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
        top = part(top_seq_modules[3], top, i+1)

    bottom_seq_modules = bottom_seq['modules']

    bottom = MaxPool2D(data_format='channels_first')(input)

    for module in bottom_seq_modules[1:-1]:
        bottom = add_id(256, module, bottom)

    bottom = UpSampling2D(data_format='channels_first')(bottom)

    out = Add()([top, bottom])

    return add_id(256, b, out)

def half(tobj, input):
    top = tobj['modules'][1]

    seq = part(top, input, 1)

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
    print("Saving h5py file")
    model.save('vrn-unguided-keras.h5')