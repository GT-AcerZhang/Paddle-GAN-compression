from layers import *
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import InstanceNorm
from paddle.nn.layer import Leaky_ReLU, ReLU, Pad2D
from .modules import MobileResnetBlock

class MobileResnetGenerator(fluid.dygraph.Layer):
    def __init__ (self, input_channel, output_nc, ngf, norm_layer=InstanceNorm, dropout_rate=0, n_blocks=9, padding_type='reflect'):
        super(MobileResnetGenerator, self).__init__()
        use_bias = norm_layer == InstanceNorm
        self.pad1 = Pad2D(paddings=[3, 3, 3, 3], mode="reflect")

        self.conv0 = conv2d(
            num_channels=input_channel,
            num_filters=ngf,
            filter_size=7,
            stride=1,
            padding=0,
            norm_layer=norm_layer,
            use_bias=use_bias)

        n_downsampling = 2

        #self.conv1 = fluid.dygraph.LayerList()
        #for i in range(n_downsampling):
        #    mult = 2 ** i
        #    self.conv1.append([conv2d(num_channels = ngf * mult, num_filters = ngf * mult * 2, filter_size = 3, stride = 2, padding=1, use_bias=use_bias)])

        self.conv1 = []
        for i in range(n_downsampling):
            mult = 2 ** i
            conv_bn_layer = self.add_sublayer("conv1_%d" % (i+1), conv2d(num_channels = ngf * mult, num_filters = ngf * mult * 2, filter_size = 3, stride = 2, padding=1, norm_layer=norm_layer, use_bias=use_bias))

            self.conv1.append(conv_bn_layer)

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        self.block1 = [] 
        self.block2 = [] 
        self.block3 = [] 
        for i in range(n_blocks1):
            block = self.add_sublayer("block1_%d" % (i+1), MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout_rate=dropout_rate, use_bias=use_bias))
            self.block1.append(block)

        for i in range(n_blocks2):
            block = self.add_sublayer("block2_%d" % (i+1), MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout_rate=dropout_rate, use_bias=use_bias))
            self.block2.append(block)

        for i in range(n_blocks3):
            block = self.add_sublayer("block3_%d" % (i+1), MobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout_rate=dropout_rate, use_bias=use_bias))
            self.block3.append(block)
        
        
        mult = 2 ** n_downsampling
        self.deconv0 = DeConv2D(
            num_channels=ngf * mult,
            num_filters=int(ngf * mult / 2),
            filter_size=3,
            stride=2,
            padding=[1, 1],
            outpadding=[0, 1, 0, 1],
            use_bias=use_bias
            )

        mult = 2 ** (n_downsampling - 1)

        self.deconv1 = DeConv2D(
            num_channels=ngf * mult,
            num_filters=int(ngf * mult / 2),
            filter_size=3,
            stride=2,
            padding=[1, 1],
            outpadding=[0, 1, 0, 1], 
            use_bias=use_bias
            )

        self.pad2 = Pad2D(paddings=[3, 3, 3, 3], mode="reflect")
        self.conv3 = conv2d(
            num_channels=ngf,
            num_filters=output_nc,
            filter_size=7,
            stride=1,
            padding=0,
            relu=False,
            norm=False,
            use_bias=True)

    def forward(self, inputs):
        #pad_input = fluid.layers.pad2d(inputs, [3, 3, 3, 3], mode="reflect")
        pad_input = self.pad1(inputs)
        print(type(pad_input), pad_input.shape)
        y = self.conv0(pad_input)
        print(y.shape)
        for conv1 in self.conv1:
            y = conv1(y)
            print(y.shape)
        for block in self.block1:
            y = block(y)
            print(y.shape)
        for block in self.block2:
            y = block(y)
            print(y.shape)
        for block in self.block3:
            y = block(y)
            print(y.shape)


        y = self.deconv0(y)
        print(y.shape)
        y = self.deconv1(y)
        print(y.shape)
        #y = fluid.layers.pad2d(y,[3,3,3,3],mode="reflect")
        print(y.shape)
        y = self.pad2(y)
        print(y.shape)
        y = self.conv3(y)
        print(y.shape)
        y = fluid.layers.tanh(y)
        return y

