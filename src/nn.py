"""
This module provides a CNN auto-encoder.
The model can be adjusted, such that 0-3 pooling operations will be performed.
"""
import numpy as np
from torch import nn


class MaxGridPool(nn.Module):
    """
    This class enables max pooling relative to the featuremaps size.
    Its counterpart is the :class:`src.nn.UpsampleGrid` class.
    """

    def __init__(self, s_in, n_bins):
        """
        :param s_in: width/ height of feature map
        :param n_bin: total number of bins to be extracted (has to be quadratic)
        """
        super().__init__()
        size = int(s_in / np.sqrt(n_bins))
        self.pooling = nn.MaxPool2d(size, stride=size, padding=0)

    def forward(self, x):
        """
        Max pooling relative to feature map height and width.

        :param x: batch of feature maps
        :return: feature maps reduced to the number of bins defined at :func:`src.nn.MaxGridPool.__init__`
        """
        return self.pooling(x)


class UpsampleGrid(nn.Module):
    """
    This class enables upsampling relative to the featuremaps size.
    Its counterpart is the :class:`src.nn.MaxGridPool` class.
    """

    def __init__(self, s_in, n_bin):
        """
        :param s_in: width/ height of initial feature map
        :param n_bin: total number of previously extracted bins (has to be quadratic)
        """
        super().__init__()
        size = int(s_in / np.sqrt(n_bin))
        self.upsample = nn.Upsample(scale_factor=size, mode='nearest')

    def forward(self, x):
        """
        :param x: batcg of feature maps
        :return: upsampled feature maps
        """
        return self.upsample(x)


class Reshape(nn.Module):
    """
    This class adapts the reshape operation to the nn.Module interface.
    """

    def __init__(self, shape_out):
        """
        :param shape_out: new shape of tensors
        """
        super().__init__()
        self.shape_out = shape_out

    def forward(self, x):
        """
        Reshaping the tensor.

        :param x: any tensor that matches the size of the initialized shape
        :return: reshaped tensor
        """
        return x.reshape(x.shape[0], *self.shape_out)


class ConvNet(nn.Module):
    """
    This class defines the auto-encoder model. Up to 3 pooling operations can be performed.

    .. note::
        Increasing the number of pooling operations will increase the depth of encoder and decoder networks.
    """

    def __init__(self, c_in=1, c_out=1, img_h=64, img_w=64, act='LReLU', n_pooling=0):
        """
        :param c_in: number of input channels/ feature maps
        :param c_out: number of ouput channels/ feature maps
        :param img_h: height of images
        :param img_w: width of images
        :param act: string indication whether to use LReLU or ReLU for hidden layers
        :param n_pooling: number of pooling operations that will be performed
        """
        assert img_h >= 32 and img_w == img_h, "images aught to be at least 32x32"
        super().__init__()
        self.n_pooling = n_pooling
        self.img_w = img_w
        self.img_h = img_h
        self.c_in = c_in
        self.c_out = c_out
        # convolution & pooling
        layer_conv = lambda l_in, l_out: nn.Sequential(nn.Conv2d(l_in, l_out, 3, padding=1), 
                                                       nn.LeakyReLU() if act == 'LReLU' else nn.ReLU(),
                                                       nn.BatchNorm2d(l_out))
        layer_deconv = lambda l_in, l_out: nn.Sequential(nn.ConvTranspose2d(l_in, l_out, 3, padding=1),
                                                         nn.LeakyReLU() if act == 'LReLU' else nn.ReLU(),
                                                         nn.BatchNorm2d(l_out))
        self.conv_1 = nn.Sequential(layer_conv(c_in, 64),
                                    layer_conv(64, 128))
        if self.n_pooling > 0:
            self.pool_1 = nn.MaxPool2d(5, stride=2, padding=2)
            self.conv_2 = nn.Sequential(layer_conv(128, 128), 
                                        layer_conv(128, 256))
            if self.n_pooling > 1:
                self.pool_2 = nn.MaxPool2d(5, stride=2, padding=2)
                self.conv_3 = nn.Sequential(layer_conv(256, 256), 
                                            layer_conv(256, 512))
                if self.n_pooling > 2:
                    coeff_ds = 1.0 / self.pool_1.stride / self.pool_2.stride
                    assert (img_h) % 1.0 == 0 and (img_w) % 1.0 == 0, \
                           f"image height and width aught to be a multiple of {int(1./coeff_ds)}"
                    h_fmap = int(img_h * coeff_ds)
                    n_bins = 16**2
                    #self.pool_3 = MaxGridPool(h_fmap, n_bins)
                    self.pool_3 = nn.MaxPool2d(5, stride=2, padding=2)
                    self.encoder = nn.Sequential(layer_conv(512, 512),
                                                 layer_conv(512, 1024))
                    self.decoder = nn.Sequential(layer_deconv(1024, 1024),
                                                 layer_deconv(1024, 512))
                    # # linear encoder & decoder after third time pooling
                    # n_features = 512
                    # d = n_features * n_bins
                    # self.reshape_enc = Reshape(shape_out=(d,))
                    # layer_lin = lambda f_in, f_out: nn.Sequential(nn.Linear(f_in, f_out), 
                    #                                               nn.LeakyReLU() if act == 'LReLU' else nn.ReLU())
                    # self.encoder = nn.Sequential(layer_lin(d, 1024), 
                    #                              #layer_lin(1024, 1024), nn.Dropout(0.5), 
                    #                              layer_lin(1024, 1024))
                    # self.decoder = nn.Sequential(#layer_lin(1024, 1024), nn.Dropout(0.5), 
                    #                              layer_lin(1024, 1024), 
                    #                              layer_lin(1024, d))
                    # self.reshape_dec = Reshape(shape_out=(n_features, int(np.sqrt(n_bins)), int(np.sqrt(n_bins))))
        # deconvolution & unpooling
        if self.n_pooling > 2:
            #self.unpool_3 = UpsampleGrid(h_fmap, n_bins)
            self.unpool_3 = nn.Upsample(scale_factor=2, mode='nearest')
        if self.n_pooling > 1:
            self.deconv_3 = nn.Sequential(layer_deconv(512, 512), layer_deconv(512, 256))
            self.unpool_4 = nn.Upsample(scale_factor=2, mode='nearest')
        if self.n_pooling > 0:
            self.deconv_4 = nn.Sequential(layer_deconv(256, 256), layer_deconv(256, 128))
            self.unpool_5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv_5 = nn.Sequential(layer_deconv(128, 128), layer_deconv(128, 64))
        # output layer
        self.output_layer = nn.ConvTranspose2d(64, c_out, 3, padding=1)

    def conv(self, x):
        """
        Convolution (/ cross correlation) part of the encoder.

        :param x: input image
        :return: convolved (/ cross correlated) feature maps
        """
        y = self.conv_1(x)
        if self.n_pooling > 0:
            y = self.pool_1(y)
            y = self.conv_2(y)
            if self.n_pooling > 1:
                y = self.pool_2(y)
                y = self.conv_3(y)
                if self.n_pooling > 2:
                    y = self.pool_3(y)
        return y

    def deconv(self, x):
        """
        The transposed convolution (/ cross correlation).

        :param x: neural code
        :return: feature maps for the output layer
        """
        y = x
        if self.n_pooling > 2:
            y = self.unpool_3(y)
        if self.n_pooling > 1:
            y = self.deconv_3(y)
            y = self.unpool_4(y)
        if self.n_pooling > 0:
            y = self.deconv_4(y)
            y = self.unpool_5(y)
        y = self.deconv_5(y)
        return y

    def encode(self, x):
        """
        Encoding the input image to a neural code.

        :param x: input image
        :return: neural code
        """
        y = self.conv(x)
        if self.n_pooling > 2:
            #y = self.reshape_enc(y)
            y = self.encoder(y)
        return y

    def decode(self, x):
        """
        Decoding the the input image from a neural code.

        :param x: neural code
        :return: feature maps for the output layer
        """
        y = x
        if self.n_pooling > 2:
            y = self.decoder(x)
            #y = self.reshape_dec(y)
        y = self.deconv(y)
        return y

    def forward(self, x):
        """
        The models forward pass, embodying encoding, decoding and final activations of the output layer.

        :param x: input image
        :return: predicted image
        """
        if self.n_pooling > 2:
            assert x.shape[-2] == self.img_h and self.img_w == x.shape[-1], \
                   f'only {self.img_h}x{self.img_w}images allowed for this model'
        y = self.encode(x)
        y = self.decode(y)
        y = self.output_layer(y)
        return y
