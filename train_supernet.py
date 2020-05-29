import paddle.fluid as fluid
from distiller.resnet_distiller import ResnetDistiller
from supernet.resnet_supernet import ResnetSupernet
from data_reader import data_reader
import argparse
import ast


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
        parser.add_argument('--crop_size', type=int, default=256, help="")
        parser.add_argument('--crop_type', type=str, default='Centor', help="")
##############
        parser.add_argument('--real_stat_path', type=str, default='real_stat/horse2zebra_B.npz', help="")
        parser.add_argument('--recon_loss_mode', type=str, default='l1', choices=['l1', 'l2'], help="")
        parser.add_argument('--gan_loss_mode', type=str, default='hinge', help="")
        parser.add_argument('--lambda_gan', type=float, default=1, help="")
        parser.add_argument('--lambda_distill', type=float, default=1, help="")
        parser.add_argument('--lambda_recon', type=float, default=100, help="")
        parser.add_argument('--teacher_ngf', type=int, default=64, help="")
        parser.add_argument('--student_ngf', type=int, default=32, help="")  ### supernet
        #parser.add_argument('--student_ngf', type=int, default=48, help="")  ### distiller
        parser.add_argument('--pretrained_ngf', type=int, default=64, help="")
        parser.add_argument('--teacher_netG', type=str, default='mobile_resnet_9blocks', help="")
        #parser.add_argument('--student_netG', type=str, default='mobile_resnet_9blocks', help="")
        parser.add_argument('--student_netG', type=str, default='super_mobile_resnet_9blocks', help="")
        parser.add_argument('--pretrained_netG', type=str, default='mobile_resnet_9blocks', help="")
        parser.add_argument('--student_dropout_rate', type=float, default=0, help="")
        parser.add_argument('--teacher_dropout_rate', type=float, default=0, help="")
        parser.add_argument('--restore_teacher_G_path', type=str, default='./output/0_tea_netG', help="")
        parser.add_argument('--restore_pretrained_G_path', type=str, default='./output/0_tea_netG', help="")
        parser.add_argument('--restore_student_G_path', type=str, default=None, help="")
        parser.add_argument('--restore_A_path', type=str, default=None, help="")
        parser.add_argument('--restore_O_path', type=str, default=None, help="")
        parser.add_argument('--restore_G_optimizer_path', type=str, default=None, help="")
        parser.add_argument('--restore_D_optimizer_path', type=str, default=None, help="")
###############
        parser.add_argument('--input_nc', type=int, default=3, help="")
        parser.add_argument('--output_nc', type=int, default=3, help="")
        parser.add_argument('--norm_type', type=str, default='instance', help="")
        parser.add_argument('--save_dir', type=str, default='./output', help="")
        #parser.add_argument('--task', type=str, default='distiller', help="")
        parser.add_argument('--task', type=str, default='supernet', help="")
        parser.add_argument('--ndf', type=int, default=128, help= "")
        parser.add_argument('--n_layer_D', type=int, default=3, help="")
        parser.add_argument('--lr', type=float, default=2e-4, help="")
        parser.add_argument('--beta1', type=float, default=0.5, help="")
        parser.add_argument('--direction', type=str, default='AtoB', help="")
        parser.add_argument('--netD', type=str, default='n_layers', help="")
###############
        parser.add_argument('--config_set', type=str, default='channels-32', help="")
        parser.add_argument('--config_str', type=str, default=None, help="")
        
        args = parser.parse_args()
        return args

args = Args().parse()
print(args)

def create_data():
    reader = data_reader(args)
    A_reader, B_reader, A_reader_test, B_reader_test, batch_num, a_id2name, b_id2name = reader.make_data()
    A_loader = fluid.io.DataLoader.from_generator(
        capacity=4,
        iterable=True,
        use_double_buffer=True)

    B_loader = fluid.io.DataLoader.from_generator(
        capacity=4,
        iterable=True,
        use_double_buffer=True)

    A_loader.set_batch_generator(
        A_reader,
        places=fluid.CUDAPlace(0) 
        if args.use_gpu else fluid.cpu_places()) ### fluid.cuda_places()
    B_loader.set_batch_generator(
        B_reader,
        places=fluid.CUDAPlace(0)
        if args.use_gpu else fluid.cpu_places()) ### fluid.cuda_places()
    return A_loader, B_loader


if __name__ == '__main__':
    with fluid.dygraph.guard():
        #model = ResnetDistiller(args)
        model = ResnetSupernet(args)
        model.setup()

        A_loader, B_loader = create_data()

        for epoch_id in range(args.epoch):
            for data_A, data_B in zip(A_loader(), B_loader()):
                model.set_input(data_A, data_B)
                model.optimize_parameter()
                print(model.get_current_loss())


