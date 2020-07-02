import argparse
import os

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import InstanceNorm, Conv2D, Conv2DTranspose
from configs import decode_config
from utils.util import load_network

def transfer_conv(m1, m2):
    param_w = m1.parameters()[0]
    c1, c2, kh, kw = m2.parameters()[0].shape
    m2.parameters()[0].set_value(param_w.numpy()[:c1, :c2, :kh, :kw])
    if len(m2.parameters()) == 2:
        c = m2.parameters()[1].shape[0]
        m2.parameters()[1].set_value(m1.parameters()[1].numpy()[:c])
    
def transfer_weight(netA, netB):
    assert len(netA.sublayers()) == len(netB.sublayers())
    for (nameA, sublayerA), (nameB, sublayerB) in zip(netA.named_sublayers(), netB.named_sublayers()):
        assert type(sublayerA) == type(sublayerB)
        assert nameA == nameB
        if isinstance(sublayerA, (Conv2D, Conv2DTranspose)):
            transfer_conv(sublayerA, sublayerB)

def main(cfgs):
    fluid.enable_imperative()
    config = decode_config(args.config_str)
    if args.model == 'mobile_resnet':
        from model.mobile_generator import MobileResnetGenerator as SuperModel
        from model.sub_mobile_generator import SubMobileResnetGenerator as SubModel
        input_nc, output_nc = args.input_nc, args.output_nc
        super_model = SuperModel(input_nc, output_nc, ngf=args.ngf, norm_layer=InstanceNorm, n_blocks=9)
        sub_model = SubModel(input_nc, output_nc, config=config, norm_layer=InstanceNorm, n_blocks=9)
    else:
        raise NotImplementedError

    load_network(super_model, args.input_path)
    transfer_weight(super_model, sub_model)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_path = os.path.join(args.output_path, 'final_net')
    fluid.save_dygraph(sub_model.state_dict(), save_path)
    print('Successfully export the subnet at [%s].' % save_path)

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, default='mobile_resnet', choices=['mobile_resnet'], help='specify the model type you want to export')
    parser.add_argument('--ngf', type=int, default=48, help='the base number of filters of the generator')
    parser.add_argument('--input_path', type=str, required=True, help='the input model path')
    parser.add_argument('--output_path', type=str, required=True, help='the path to the exported model')
    parser.add_argument('--config_str', type=str, required=True,
                        help='the configuration string for a specific subnet in the supernet')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    cfgs = parser.parse_args()
    main(cfgs)  

