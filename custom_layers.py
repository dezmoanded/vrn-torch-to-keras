from keras.layers import Conv2D, Layer, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np
import tensorflow as tf

class Conv(Conv2D):
    def __init__(self, output_dim, size, strides=1, padding='valid', tobj=None, **kwargs):
        self.output_dim = output_dim
        self.size = size
        self.tobj = tobj

        super(Conv, self).__init__(output_dim,
                                   size,
                                   strides=strides,
                                   padding=padding,
                                   data_format='channels_first')

    def reorder(self, tensor4):
        tensor4 = np.swapaxes(tensor4, 0, 2)
        tensor4 = np.swapaxes(tensor4, 1, 3)
        tensor4 = np.swapaxes(tensor4, 2, 3)
        return tensor4

    def build(self, input_shape):
        super(Conv, self).build(input_shape)
        if self.tobj is not None:
            weight = self.reorder(self.tobj['weight'])
            self.set_weights((weight, self.tobj['bias']))

    def call(self, inputs):
        out = inputs
        if self.size == 3:
            out = ZeroPadding2D(padding=1, data_format='channels_first')(inputs)
        return super(Conv, self).call(out)

    def compute_output_shape(self, input_shape):
        if self.size == 3:
            return input_shape
        return super(Conv, self).compute_output_shape(input_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'size': self.size}
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BatchNorm(BatchNormalization):
    def __init__(self, tobj=None, **kwargs):
        self.tobj = tobj
        super(BatchNorm, self).__init__(axis=1,
                                        momentum=.1,
                                        epsilon=1e-5)
    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        if self.tobj is not None:
            super(BatchNorm, self).set_weights((self.tobj['weight'],
                                                self.tobj['bias'],
                                                self.tobj['running_mean'],
                                                self.tobj['running_var']))

    def call(self, inputs):
        return super(BatchNorm, self).call(inputs, training=True)

    def compute_output_shape(self, input_shape):
        return super(BatchNorm, self).compute_output_shape(input_shape)

class UpSamplingBilinear(Layer):
    def __init__(self, scale=4, **kwargs):
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

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(UpSamplingBilinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))