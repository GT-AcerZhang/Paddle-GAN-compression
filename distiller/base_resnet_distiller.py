import os
import numpy as np
import itertools
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.base import to_variable
from model.super_modules import SuperConv2D
from model import loss
from model import network
from utils import util, optimization
from data_loader import create_eval_data
from metric.inception import InceptionV3

class BaseResnetDistiller(object):
    def __init__(self, args):
        super(BaseResnetDistiller, self).__init__()
        self.args = args
        self.loss_names = ['G_gan', 'G_distill', 'G_recon', 'D_fake', 'D_real']
        #self.optimizers = []
        self.image_paths = []
        #self.visual_names = ['real_A', 'Sfake_B', 'Tfake_B', 'real_B']
        self.model_names = ['netG_student', 'netG_teacher', 'netD']
        self.netG_teacher = network.define_G(args.input_nc, args.output_nc, args.teacher_ngf, args.teacher_netG, args.norm_type, args.teacher_dropout_rate)
        self.netG_student = network.define_G(args.input_nc, args.output_nc, args.student_ngf, args.student_netG, args.norm_type, args.student_dropout_rate)
        if args.task == 'distiller':
            self.netG_pretrained = network.define_G(args.input_nc, args.output_nc, args.pretrained_ngf, args.pretrained_netG, args.norm_type, 0)
            print(self.netG_pretrained)

        self.netD = network.define_D(args.output_nc, args.ndf, args.netD, args.norm_type, args.n_layer_D)

        ### need to check mapping layer
        ### [9, 12, 15, 18]
        self.mapping_layers = ['model.%d' % i for i in range(9, 21, 3)] ##[12, 57, 102, 147]  ### [i for i in range(9, 21, 3)]
        self.netAs = []
        self.Tacts, self.Sacts = {}, {}

        #G_params = [self.netG_student.parameters()]
        G_params = self.netG_student.parameters()
        for i, n in enumerate(self.mapping_layers):
            ft, fs = args.teacher_ngf, args.student_ngf
            if args.task == 'distiller':
                netA = Conv2D(num_channels = fs * 4, num_filters = ft * 4, filter_size=1)
            else:
                netA = SuperConv2D(num_channels = fs * 4, num_filters = ft * 4, filter_size=1)

            #G_params.append(netA.parameters())
            G_params += netA.parameters()
            self.netAs.append(netA)
            self.loss_names.append('G_distill%d' % i)

        self.optimizer_G = optimization.Optimizer(args, parameter_list=G_params)
        self.optimizer_D = optimization.Optimizer(args, parameter_list=self.netD.parameters())
        #self.optimizer_G = fluid.optimizer.Adam(learning_rate=args.lr, beta1=args.beta1, beta2=0.999, parameter_list=itertools.chain(*G_params))
        #self.optimizer_D = fluid.optimizer.Adam(learning_rate=args.lr, beta1=args.beta1, beta2=0.999, parameter_list=self.netD.parameters())
        self.eval_dataloader = create_eval_data(args, direction=args.direction)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])
        self.is_best = False

        if args.real_stat_path:
            self.npz = np.load(args.real_stat_path)
       
        self.is_best = False

    def setup(self):
        self.load_networks()
        self.netG_teacher.eval()
        #self.save_network(0)
        #sys.exit(0)

        if self.args.lambda_distill > 0:
            def get_activation(mem, name):
                def get_output_hook(layer, input, output):
                    mem[name] = output
                return get_output_hook
                
            def add_hook(net, mem, mapping_layers):
                for idx, (n, m) in enumerate(net.named_sublayers()):
                    #if idx in mapping_layers:
                    if n in mapping_layers:
#                        print(idx, n, m)
                        m.register_forward_post_hook(get_activation(mem, n))
                        #m.register_forward_post_hook(get_activation(mem, idx))

        add_hook(self.netG_teacher, self.Tacts, self.mapping_layers)
        add_hook(self.netG_student, self.Sacts, self.mapping_layers)

    def set_input(self, input_A, input_B):
        self.real_A = input_A[0]
        self.real_B = input_B[0]
        #AtoB = self.args.direction == 'AtoB'
        #self.real_A = input['input_A' if AtoB else 'input_B']
        #self.real_B = input['input_B' if AtoB else 'input_A']
        #self.image_path = input['A_paths' if AtoB else 'B_paths']

    def set_single_input(self, input):
        self.real_A = input[0]

    def load_networks(self):
        util.load_network(self.netG_teacher, self.args.restore_teacher_G_path)
        if self.args.restore_student_G_path is not None:
            util.load_network(self.netG_student, self.args.restore_student_G_path)
        if self.args.restore_D_path is not None:
            util.load_network(self.netD, self.args.restore_D_path)
        if self.args.restore_A_path is not None:
            for i, netA in enumerate(self.netAs):
                netA_path = '%s-%d' % (self.args.restore_A_path, i)
                util.load_network(netA, netA_path)
        if self.args.restore_O_path is not None:
            util.load_optimizer(self.optimizer_G, self.args.restore_G_optimizer_path)
            util.load_optimizer(self.optimizer_D, self.args.restore_D_optimizer_path)
            

    def backward_D(self):
        #fake = to_variable(self.Sfake_B)
        #real = to_variable(self.real_B)
        fake = self.Sfake_B.detach()
        real = self.real_B.detach()

        pred_fake = self.netD(fake)
        self.loss_D_fake = loss.gan_loss(self.args.gan_loss_mode, pred_fake, False)

        pred_real = self.netD(real)
        self.loss_D_real = loss.gan_loss(self.args.gan_loss_mode, pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def calc_distill_loss(self):
        raise NotImplementedError

    def backward_G(self):
        raise NotImplementedError

    def optimize_parameter(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def save_network(self, epoch):
        save_filename = '{}_stu_netG'.format(epoch)
        save_path = os.path.join(self.args.save_dir, save_filename)
        fluid.save_dygraph(self.netG_student.state_dict(), save_path)
        save_filename = '{}_tea_netG'.format(epoch)
        save_path = os.path.join(self.args.save_dir, save_filename)
        fluid.save_dygraph(self.netG_teacher.state_dict(), save_path)

        save_filename = '{}_netD'.format(epoch)
        save_path = os.path.join(self.args.save_dir, save_filename)
        fluid.save_dygraph(self.netD.state_dict(), save_path)

        for idx, net in enumerate(self.netAs):
            save_filename = '{}_netA-{}'.format(epoch, idx)
            save_path = os.path.join(self.args.save_dir, save_filename)
            fluid.save_dygraph(net.state_dict(), save_path)

        #save_filename = '{}_optimG'.format(epoch)
        #save_path = os.path.join(self.args.save_dir, save_filename)
        #fluid.save_dygraph(self.optimizer_G.state_dict(), save_path)
        #save_filename = '{}_optimD'.format(epoch)
        #save_path = os.path.join(self.args.save_dir, save_filename)
        #fluid.save_dygraph(self.optimizer_D.state_dict(), save_path)

    def get_current_loss(self):
        loss_dict = {}
        for name in self.loss_names:
            if not hasattr(self, 'loss_' + name):
                continue
            key = name
            loss_dict[key] = float(getattr(self, 'loss_' + name))
        return loss_dict

    def get_current_lr(self):
        lr_dict = {}
        lr_dict['optim_G'] = self.optimizer_G.optimizer.current_step_lr()
        lr_dict['optim_D'] = self.optimizer_D.optimizer.current_step_lr()
        return lr_dict

    @fluid.dygraph.no_grad
    def test(self):
        self.forward()
