import paddle.fluid as fluid
from distiller.resnet_distiller import ResnetDistiller
from supernet.resnet_supernet import ResnetSupernet
from data_loader import create_data
import argparse
import ast


class Args:
    def __init__(self):
        self.isTrain = True

    def parse(self):
        parser = argparse.ArgumentParser(description=__doc__)
        
        parser.add_argument('--isTrain', type=ast.literal_eval, default=True, help="")
        parser.add_argument('--use_gpu', type=ast.literal_eval, default=True, help='Whether to use GPU in train/test model.')
        parser.add_argument('--epoch', type=int, default=400, help="")
### data
        parser.add_argument('--batch_size', type=int, default=1, help="")
        parser.add_argument('--shuffle', type=ast.literal_eval, default=True, help="")
        parser.add_argument('--dataset', type=str, default='horse2zebra', help="")
        parser.add_argument('--data_dir', type=str, default='./data', help="")
        parser.add_argument('--model_net', type=str, default='CycleGAN', help="")
        parser.add_argument('--run_test', type=ast.literal_eval, default=False, help='')
        parser.add_argument('--image_size', type=int, default=286, help="")
        parser.add_argument('--crop_size', type=int, default=256, help="")
        parser.add_argument('--max_dataset_size', type=int, default=1333, help="")
        parser.add_argument('--crop_type', type=str, default='Random', help="")
        parser.add_argument('--nepochs', type=int, default=200, help="")
        parser.add_argument('--nepochs_decay', type=int, default=200, help="")
##############
        parser.add_argument('--real_stat_path', type=str, default='real_stat/horse2zebra_B.npz', help="")
        parser.add_argument('--recon_loss_mode', type=str, default='l1', choices=['l1', 'l2'], help="")
        parser.add_argument('--gan_loss_mode', type=str, default='lsgan', help="")
        parser.add_argument('--lambda_gan', type=float, default=1, help="")
        parser.add_argument('--lambda_distill', type=float, default=0.01, help="")
        parser.add_argument('--lambda_recon', type=float, default=10.0, help="")
        parser.add_argument('--teacher_ngf', type=int, default=64, help="")
        parser.add_argument('--student_ngf', type=int, default=32, help="")  ### supernet
        parser.add_argument('--teacher_netG', type=str, default='mobile_resnet_9blocks', help="")
        parser.add_argument('--student_netG', type=str, default='super_mobile_resnet_9blocks', help="")
        parser.add_argument('--student_dropout_rate', type=float, default=0, help="")
        parser.add_argument('--teacher_dropout_rate', type=float, default=0, help="")
        #parser.add_argument('--restore_teacher_G_path', type=str, default='./pth_mobile_weight/numpy_latest_net_G_A.pkl', help="")
        #parser.add_argument('--restore_student_G_path', type=str, default='./pth_mobile_weight/numpy_latest_net_G.pth', help="")
        #parser.add_argument('--restore_D_path', type=str, default='./pth_mobile_weight/numpy_latest_net_D.pth', help="")
        #parser.add_argument('--restore_A_path', type=str, default='./pth_mobile_weight/numpy_0_net_A', help="")
        parser.add_argument('--restore_teacher_G_path', type=str, default='./output_mobile_V3_V1/199_net_G_A', help="")
        parser.add_argument('--restore_student_G_path', type=str, default='./output_distiller_debug/199_stu_netG', help="")
        parser.add_argument('--restore_D_path', type=str, default='./output_distiller_debug/199_netD', help="")
        parser.add_argument('--restore_A_path', type=str, default=None, help="")
        parser.add_argument('--restore_O_path', type=str, default=None, help="")
###############
        parser.add_argument('--input_nc', type=int, default=3, help="")
        parser.add_argument('--output_nc', type=int, default=3, help="")
        parser.add_argument('--norm_type', type=str, default='instance', help="")
        parser.add_argument('--save_dir', type=str, default='./output_supernet_V3_V1', help="")
        parser.add_argument('--task', type=str, default='supernet', help="")
        parser.add_argument('--ndf', type=int, default=64, help= "")
        parser.add_argument('--n_layer_D', type=int, default=3, help="")
        parser.add_argument('--lr', type=float, default=2e-4, help="")
        parser.add_argument('--beta1', type=float, default=0.5, help="")
        parser.add_argument('--direction', type=str, default='AtoB', help="")
        parser.add_argument('--netD', type=str, default='n_layers', help="")
        parser.add_argument('--scheduler', type=str, default='linear', help="")
        parser.add_argument('--step_per_epoch', type=int, default=1334, help="")
        parser.add_argument('--inception_model', type=str, default='metric/params_inceptionV3', help="")
###############
        parser.add_argument('--config_set', type=str, default=None, help="")
        #parser.add_argument('--config_set', type=str, default='channels-32', help="")
        parser.add_argument('--config_str', type=str, default=None, help="")
        
        args = parser.parse_args()
        return args

args = Args().parse()
print(args)



if __name__ == '__main__':
    fluid.enable_imperative() 
    model = ResnetSupernet(args)
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

            if batch_id % 100 == 0:
                message = '(epoch: %d, batch: %d\t' % (epoch_id, batch_id)
                for k, v in model.get_current_loss().items():
                    message += '%s: %.3f ' % (k, v)
                for k, v in model.get_current_lr().items():
                    message += '%s: %f ' % (k, v)
                print(message)
                #print(epoch_id, batch_id, model.get_current_loss())
                #print(epoch_id, batch_id, model.get_current_lr())
        model.evaluate_model(epoch_id)
        if epoch_id % 10 == 0 or epoch_id == (args.epoch-1):
            model.save_network(epoch_id)





