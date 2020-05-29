import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, BatchNorm, InstanceNorm, Dropout
from paddle.nn.layer import Leaky_ReLU, ReLU, Pad2D

__all__ = ['SeparableConv2D', 'MobileResnetBlock', 'ResnetBlock']

class SeparableConv2D(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, filter_size, stride=1, padding=0, norm_layer=InstanceNorm, use_bias=True, scale_factor=1, stddev=0.02, use_cudnn=True):
        super(SeparableConv2D, self).__init__()
        if use_bias == False:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.ParamAttr(initializer=fluid.initializer.Constant(0.0))

        self.conv_sep = Conv2D(
            num_channels=num_channels,
            num_filters=num_channels * scale_factor,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            groups = num_channels,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(loc=0.0,scale=stddev)),
            bias_attr=con_bias_attr)

        if norm_layer == InstanceNorm:
            self.norm = InstanceNorm(
                num_channels=num_channels * scale_factor,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(1.0), trainable=False),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0), trainable=False),
                )
        elif norm_layer == BatchNorm:
            self.norm = BatchNorm(
                num_channels=num_channels * scale_factor,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NormalInitializer(1.0,0.02)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)),
                )
        else:
            raise NotImplementedError


        self.conv_out = Conv2D(
            num_channels=num_channels * scale_factor,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(loc=0.0,scale=stddev)),
            bias_attr=con_bias_attr)
    
    def forward(self,inputs):
        conv = self.conv_sep(inputs)
        conv = self.norm(conv)
        conv = self.conv_out(conv)
        return conv


class MobileResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(MobileResnetBlock, self).__init__()
        self.padding_type = padding_type
        self.dropout_rate = dropout_rate
        self.conv_block = fluid.dygraph.LayerList([])

        p = 0
        if self.padding_type == 'reflect':
            self.conv_block.extend([Pad2D(paddings=[1,1,1,1], mode='reflect')])
            #self.pad1 = Pad2D(paddings=[1,1,1,1], mode='reflect')
        elif self.padding_type == 'replicate':
            self.conv_block.extend([Pad2D(inputs, paddings=[1,1,1,1], mode='edge')])
            #self.pad1 = Pad2D(inputs, paddings=[1,1,1,1], mode='edge')
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % self.padding_type)

        self.conv_block.extend([SeparableConv2D(num_channels=dim, num_filters=dim, filter_size=3, padding=p, stride=1), norm_layer(dim), ReLU()])

        self.conv_block.extend(Dropout(p=self.dropout_rate))
        #self.sep_conv1 = SeparableConv2D(num_channels=dim, num_filters=dim, norm_layer=norm_layer, filter_size=3, padding=p, stride=1)
        #self.norm1 = norm_layer(dim)
        #self.relu = ReLU()
        #self.dropout = Dropout(p=self.dropout_rate)

        if self.padding_type == 'reflect':
            self.conv_block.extend([Pad2D(paddings=[1,1,1,1], mode='reflect')])
            #self.pad2 = Pad2D(paddings=[1,1,1,1], mode='reflect')
        elif self.padding_type == 'replicate':
            self.conv_block.extend([Pad2D(inputs, paddings=[1,1,1,1], mode='edge')])
            #self.pad2 = Pad2D(inputs, paddings=[1,1,1,1], mode='edge')
        elif self.padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % self.padding_type)

        self.conv_block.extend([SeparableConv2D(num_channels=dim, num_filters=dim, filter_size=3, padding=p, stride=1), norm_layer(dim)])
        #self.sep_conv2 = SeparableConv2D(num_channels=dim, num_filters=dim, norm_layer=norm_layer, filter_size=3, padding=p, stride=1)
        #self.norm2 = norm_layer(dim)

    def forward(self, inputs):
        #if self.padding_type == 'zero':
        #    conv = inputs
        #else:
        #    conv = self.pad1(inputs)

        #conv = self.sep_conv1(conv)
        #conv = self.norm1(conv)
        #conv = self.relu(conv)
        ##conv = fluid.layers.relu(conv)
        #
        #conv = self.dropout(conv)
        ##conv = fluid.layers.dropout(conv, self.dropout_rate)

        ##if self.padding_type == 'reflect':
        ##    conv = fluid.layers.pad2d(conv, paddings=[1,1,1,1], mode='reflect')
        ##elif self.padding_type == 'replicate':
        ##    conv = fluid.layers.pad2d(conv, paddings=[1,1,1,1], mode='edge')
        #if self.padding_type == 'zero':
        #    conv = conv
        #else:
        #    conv = self.pad2(conv)

        #conv = self.sep_conv2(conv)
        #conv = self.norm2(conv)

        out = inputs + self.conv_block(inputs)
        return out
        
class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self,
        dim,
        use_bias=False):
        super(ResnetBlock,self).__init__()

        self.pad1 = Pad2D(paddings=[1,1,1,1], mode='reflect')
        self.conv0 = conv2d(
            num_channels=dim,
            num_filters=dim,
            filter_size=3,
            stride=1,
            stddev=0.02,
            use_bias=False)
        self.dropout = Dropout()
        self.pad2 = Pad2D(paddings=[1,1,1,1], mode='reflect')
        self.conv1 = conv2d(
            num_channels=dim,
            num_filters=dim,
            filter_size=3,
            stride=1,
            stddev=0.02,
            relu=False,
            use_bias=False)
        self.dim = dim

    def forward(self,inputs):
        #out_res = fluid.layers.pad2d(inputs, [1, 1, 1, 1], mode="reflect")
        out_res = self.pad1(inputs)
        out_res = self.conv0(out_res)
        
        if self.use_dropout:
            out_res = self.dropout(out_res)
            #out_res = fluid.layers.dropout(out_res, dropout_rate=0.5)

        #out_res = fluid.layers.pad2d(out_res, [1, 1, 1, 1], mode="reflect")
        out_res = self.pad2(out_res)
        out_res = self.conv1(out_res)
        return out_res + inputs
