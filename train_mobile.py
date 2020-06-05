import os
import paddle.fluid as fluid
from model.cycle_gan_model import CycleGAN
from data_loader import create_data
import argparse
import ast

os.environ['FLAGS_cudnn_exhaustive_search'] = 'False'
os.environ['FLAGS_cudnn_deterministic'] = 'True'

class Args:
    def __init__(self):
        self.isTrain = True

    def parse(self):
        parser = argparse.ArgumentParser(description=__doc__)
        
        parser.add_argument('--isTrain', type=ast.literal_eval, default=True, help="")
        parser.add_argument('--use_gpu', type=ast.literal_eval, default=True, help='Whether to use GPU in train/test model.')
        parser.add_argument('--epoch', type=int, default=200, help="")
### data
        parser.add_argument('--batch_size', type=int, default=1, help="")
        parser.add_argument('--shuffle', type=ast.literal_eval, default=True, help="")
        parser.add_argument('--dataset', type=str, default='horse2zebra', help="")
        parser.add_argument('--data_dir', type=str, default='./data', help="")
        parser.add_argument('--model_net', type=str, default='CycleGAN', help="")
        parser.add_argument('--run_test', type=ast.literal_eval, default=False, help='')
        parser.add_argument('--image_size', type=int, default=286, help="")
        parser.add_argument('--max_dataset_size', type=int, default=1334, help="")
        parser.add_argument('--crop_size', type=int, default=256, help="")
        parser.add_argument('--crop_type', type=str, default='Random', help="")
##############
        parser.add_argument('--real_stat_A_path', type=str, default='real_stat/horse2zebra_A.npz', help="")
        parser.add_argument('--real_stat_B_path', type=str, default='real_stat/horse2zebra_B.npz', help="")
        parser.add_argument('--recon_loss_mode', type=str, default='l1', choices=['l1', 'l2'], help="")
        parser.add_argument('--gan_loss_mode', type=str, default='lsgan', help="")
        parser.add_argument('--lambda_identity', type=float, default=0.5, help="")
        parser.add_argument('--lambda_A', type=float, default=10.0, help="")
        parser.add_argument('--lambda_B', type=float, default=10.0, help="")
        parser.add_argument('--ngf', type=int, default=64, help="")
        parser.add_argument('--pretrained_ngf', type=int, default=64, help="")
        parser.add_argument('--netG', type=str, default='mobile_resnet_9blocks', help="")
        parser.add_argument('--dropout_rate', type=float, default=0, help="")
###############
        parser.add_argument('--input_nc', type=int, default=3, help="")
        parser.add_argument('--output_nc', type=int, default=3, help="")
        parser.add_argument('--norm_type', type=str, default='instance', help="")
        parser.add_argument('--save_dir', type=str, default='./output_mobile_V3', help="")
        parser.add_argument('--ndf', type=int, default=64, help= "")
        parser.add_argument('--n_layer_D', type=int, default=3, help="")
        parser.add_argument('--lr', type=float, default=2e-4, help="")
        parser.add_argument('--beta1', type=float, default=0.5, help="")
        parser.add_argument('--direction', type=str, default='AtoB', help="")
        parser.add_argument('--netD', type=str, default='n_layers', help="")
        parser.add_argument('--nepochs', type=int, default=100, help="")
        parser.add_argument('--nepochs_decay', type=int, default=100, help="")
        parser.add_argument('--scheduler', type=str, default='linear', help="")
        parser.add_argument('--pool_size', type=int, default=50, help="")
        parser.add_argument('--step_per_epoch', type=int, default=1333, help="")
        parser.add_argument('--restore_G_A_path', type=str, default=None, help="")
        parser.add_argument('--restore_G_B_path', type=str, default=None, help="")
        parser.add_argument('--restore_D_A_path', type=str, default=None, help="")
        parser.add_argument('--restore_D_B_path', type=str, default=None, help="")
        parser.add_argument('--inception_model', type=str, default='metric/params_inceptionV3', help="")
        #parser.add_argument('--restore_G_A_path', type=str, default='./output_mobile/numpy_0_net_G_A.pth', help="")
        #parser.add_argument('--restore_G_B_path', type=str, default='./output_mobile/numpy_0_net_G_B.pth', help="")
        #parser.add_argument('--restore_D_A_path', type=str, default='./output_mobile/numpy_0_net_D_A.pth', help="")
        #parser.add_argument('--restore_D_B_path', type=str, default='./output_mobile/numpy_0_net_D_B.pth', help="")
        
        args = parser.parse_args()
        return args

args = Args().parse()
print(args)


if __name__ == '__main__':
    fluid.enable_imperative() 
    model = CycleGAN(args)
    model.setup()

    A_loader, B_loader = create_data(args)

    for epoch_id in range(args.epoch):
        for batch_id, (data_A, data_B) in enumerate(zip(A_loader(), B_loader())):
            #import pickle
            #import numpy as np
            #from paddle.fluid.dygraph.base import to_variable
            #data = pickle.load(open('data_.pkl', 'rb'))
            #data_A = to_variable(np.expand_dims(data['A'], axis=0))
            #data_B = to_variable(np.expand_dims(data['B'], axis=0))
            model.set_input(data_A, data_B)
            model.optimize_parameter()
            #if batch_id % 10 == 0:
            if batch_id % 100 == 0:
                message = '(epoch: %d, batch: %d\t' % (epoch_id, batch_id)
                for k, v in model.get_current_loss().items():
                    message += '%s: %.3f ' % (k, v)
                for k, v in model.get_current_lr().items():
                    message += '%s: %f ' % (k, v)
                print(message)
                #print(epoch_id, batch_id, model.get_current_loss())
                #print(epoch_id, batch_id, model.get_current_lr())
#            if batch_id == 3:
#                sys.exit(0)
        model.evaluate_mode(epoch_id)
        if epoch_id % 10 == 0 or epoch_id == (args.epoch-1):
            model.save_network(epoch_id)


