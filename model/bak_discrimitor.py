import paddle.fluid as fluid
from layers import *
from paddle.fluid.dygraph.nn import InstanceNorm

class build_discriminator(fluid.dygraph.Layer):
    def __init__(self, input_channel, ndf, n_layers=3, norm_layer=InstanceNorm):
        super(build_discriminator, self).__init__()
        
        use_bias = norm_layer == InstanceNorm

        self.conv_first = conv2d(
            num_channels=input_channel,
            num_filters=ndf,
            filter_size=4,
            stride=2,
            stddev=0.02,
            padding=1,
            norm=False,
            use_bias=use_bias,
            relufactor=0.2)

        ndf_mult_prev = 1
        ndf_mult = 1
        self.conv = []
        for i in range(1, n_layers):
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** i, 8)
            conv = self.add_sublayer("conv_%d" % (i), conv2d(num_channels=ndf * ndf_mult_prev, num_filters = ndf * ndf_mult, filter_size = 4, stride=2, padding=1, use_bias=use_bias))
            self.conv.append(conv)

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)
        self.conv_next_to_last = conv2d(
            num_channels=ndf * ndf_mult_prev,
            num_filters=ndf * ndf_mult,
            filter_size=4,
            stride=1,
            stddev=0.02,
            padding=1,
            use_bias=use_bias,
            relufactor=0.2)
        self.conv_last = conv2d(
            num_channels=ndf * ndf_mult,
            num_filters=1,
            filter_size=4,
            stride=1,
            stddev=0.02,
            padding=1,
            norm=False,
            relu=False,
            use_bias=True)

    def forward(self,inputs):
        y = self.conv_first(inputs)
        for conv in self.conv:
            y = conv(y)
        y = self.conv_next_to_last(y)
        y = self.conv_last(y)
        return y


