import itertools
import os
import numpy as np
import paddle.fluid as fluid
from data_loader import create_eval_data
from utils.image_pool import ImagePool
from model import network, loss
from metric.inception import InceptionV3
from utils import util, optimization
from metric import get_fid

class CycleGAN(fluid.dygraph.Layer):
    def __init__(self, args):
        super(CycleGAN, self).__init__()
        assert args.isTrain
        assert args.direction == 'AtoB'
        self.args = args
        self.loss_names = ['D_A', 'G_A', 'G_cycle_A', 'G_idt_A', 'D_B', 'G_B', 'G_cycle_B', 'G_idt_B']
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']

        self.netG_A = network.define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm_type, args.dropout_rate)
        self.netG_B = network.define_G(args.output_nc, args.input_nc, args.ngf, args.netG, args.norm_type, args.dropout_rate)
        self.netD_A = network.define_D(args.output_nc, args.ndf, args.netD, args.norm_type, args.n_layer_D)
        self.netD_B = network.define_D(args.input_nc, args.ndf, args.netD, args.norm_type, args.n_layer_D)

        if args.lambda_identity > 0.0:
            assert (args.input_nc == args.output_nc)
        self.fake_A_pool = ImagePool(args.pool_size)
        self.fake_B_pool = ImagePool(args.pool_size)

        #self.optimizer_G = fluid.optimizer.Adam(learning_rate=args.lr, beta1=args.beta1, beta2=0.999, parameter_list=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        #self.optimizer_D = fluid.optimizer.Adam(learning_rate=args.lr, beta1=args.beta1, beta2=0.999, parameter_list=itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()))
        #self.optimizer_G = optimization.Optimizer(args, parameter_list=itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()))
        self.optimizer_G = optimization.Optimizer(args, parameter_list=(self.netG_A.parameters() + self.netG_B.parameters()))
        ###print("=========== Net A ===============")
        ###for key in self.netD_A.parameters():
        ###    print(key.name)
        ###print("=========== Net B ===============")
        ###for key in self.netD_B.parameters():
        ###    print(key.name)
        ###print("=========== Net A+B ===============")
        ###for key in itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()):
        ###    print(key.name)
        #print("==========================")
        self.optimizer_D_A = optimization.Optimizer(args, parameter_list=self.netD_A.parameters())
        self.optimizer_D_B = optimization.Optimizer(args, parameter_list=self.netD_B.parameters())
        #self.optimizer_D = optimization.Optimizer(args, parameter_list=itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()))

        self.eval_dataloader_AtoB = create_eval_data(args, direction='AtoB')
        self.eval_dataloader_BtoA = create_eval_data(args, direction='BtoA')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception_model = InceptionV3([block_idx])

        self.best_fid_A, self.best_fid_B = 1e9, 1e9
        self.fids_A, self.fids_B = [], []
        self.is_best = False
        self.npz_A = np.load(args.real_stat_A_path)
        self.npz_B = np.load(args.real_stat_B_path)

    def set_input(self, input_A, input_B):
        self.real_A = input_A[0]
        self.real_B = input_B[0]

    def set_single_input(self, input):
        self.real_A = input[0]

    def setup(self):
        #self.save_network(0)
        #sys.exit(0)
        #self.load_network()
        self._load_net()

    def _load_net(self):
        import cPickle as pickle
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            print(net)
            path = getattr(self.args, 'restore_%s_path' % name, None)
            if path is not None:
                params = pickle.load(open(path, 'rb'))
                net.set_dict(params)

    def load_network(self):
        for name in self.model_names:
            net = getattr(self, 'net' + name, None)
            path = getattr(self.args, 'restore_%s_path' % name, None)
            if path is not None:
                util.load_network(net, path)

    def save_network(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s' % (epoch, name)
                save_path = os.path.join(self.args.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                fluid.save_dygraph(net.state_dict(), save_path)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  ## G_A(A)
        print("netG_A", np.sum(np.abs(self.real_A.numpy())), np.sum(np.abs(self.fake_B.detach().numpy())))
        self.rec_A = self.netG_B(self.fake_B)  ## G_B(G_A(A))
        print("netG_B", np.sum(np.abs(self.fake_B.numpy())), np.sum(np.abs(self.rec_A.detach().numpy())))
        self.fake_A = self.netG_B(self.real_B)  ## G_B(B)
        print("netG_B", np.sum(np.abs(self.real_B.numpy())), np.sum(np.abs(self.fake_A.detach().numpy())))
        self.rec_B = self.netG_A(self.fake_A)  ## G_A(G_B(B))
        print("netG_A", np.sum(np.abs(self.fake_A.numpy())), np.sum(np.abs(self.rec_B.detach().numpy())))

    def backward_D_basic(self, netD, real, fake):
        ### real
        pred_real = netD(real)
        loss_D_real = loss.gan_loss(self.args.gan_loss_mode, pred_real, True)
        ### fake
        pred_fake = netD(fake.detach())
        loss_D_fake = loss.gan_loss(self.args.gan_loss_mode, pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        #print("========================================= backward DA ====================================")
        #for named, sublayer in self.netG_A.named_sublayers():
        #    print(named, sublayer)
        #    if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
        #        print(sublayer.weight.stop_gradient)
        #        print(sublayer.bias.stop_gradient)
        #    if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
        #        print(sublayer.scale.stop_gradient)
        #print("=========================================================================================")
        #for named, sublayer in self.netD_A.named_sublayers():
        #    print(named, sublayer)
        #    if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
        #        print(sublayer.weight.stop_gradient)
        #        print(sublayer.bias.stop_gradient)
        #    if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
        #        print(sublayer.scale.stop_gradient)
        #print("=========================================================================================")
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        #print("========================================= backward DB ====================================")
        #for named, sublayer in self.netG_B.named_sublayers():
        #    print(named, sublayer)
        #    if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
        #        print(sublayer.weight.stop_gradient)
        #        print(sublayer.bias.stop_gradient)
        #    if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
        #        print(sublayer.scale.stop_gradient)
        #print("=========================================================================================")
        #for named, sublayer in self.netD_B.named_sublayers():
        #    print(named, sublayer)
        #    if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
        #        print(sublayer.weight.stop_gradient)
        #        print(sublayer.bias.stop_gradient)
        #    if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
        #        print(sublayer.scale.stop_gradient)
        #print("=========================================================================================")
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.args.lambda_identity
        lambda_A = self.args.lambda_A
        lambda_B = self.args.lambda_B
        print("========================================= backward G ====================================")
        for named, sublayer in self.netG_A.named_sublayers():
            print(named, sublayer)
            if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
                print(sublayer.bias.trainable) #stop_gradient)
            if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
                print(sublayer.scale.stop_gradient)
        print("=========================================================================================")
        for named, sublayer in self.netD_A.named_sublayers():
            print(named, sublayer)
            if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
                print(sublayer.weight.stop_gradient)
                print(sublayer.bias.stop_gradient)
            if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
                print(sublayer.scale.stop_gradient)
        print("=========================================================================================")
        #sys.exit(0)

        if lambda_idt > 0:
            ### identity loss G_A: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_G_idt_A = loss.recon_loss('l1', self.idt_A, self.real_B) * lambda_B * lambda_idt
            ### identity loss G_B: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_G_idt_B = loss.recon_loss('l1', self.idt_B, self.real_A) * lambda_A * lambda_idt

        else:
            self.loss_G_idt_A = 0
            self.loss_G_idt_B = 0

        ### GAN loss D_A(G_A(A))
        self.loss_G_A = loss.gan_loss(self.args.gan_loss_mode, self.netD_A(self.fake_B), True)
        ### GAN loss D_B(G_B(B))
        self.loss_G_B = loss.gan_loss(self.args.gan_loss_mode, self.netD_B(self.fake_A), True)
        ### forward cycle loss ||G_B(G_A(A)) - A||
        self.loss_G_cycle_A = loss.recon_loss('l1', self.rec_A, self.real_A) * lambda_A
        ### backward cycle loss ||G_A(G_B(B)) - B||
        self.loss_G_cycle_B = loss.recon_loss('l1', self.rec_B, self.real_B) * lambda_B
        ### combine loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_G_cycle_A + self.loss_G_cycle_B + self.loss_G_idt_A + self.loss_G_idt_B

        self.loss_G.backward()
        #print("========================================= backward G ====================================")
        #for named, sublayer in self.netG_A.named_sublayers():
        #    print(named, sublayer)
        #    if isinstance(sublayer, fluid.dygraph.nn.Conv2D) or isinstance(sublayer, fluid.dygraph.nn.Conv2DTranspose):
        #        print(sublayer.weight.stop_gradient, sublayer.weight.gradient())
        #    if isinstance(sublayer, fluid.dygraph.nn.InstanceNorm):
        #        print(sublayer.scale.stop_gradient, sublayer.scale.gradient())
        #print("=========================================================================================")

    def optimize_parameter(self):
        print('optimize_parameter')
        self.forward()
        ### TODO: DONNOT need to update params in netD, need to check it
        self.set_stop_gradient([self.netD_A, self.netD_B], True)
        #self.set_stop_gradient([self.netG_A, self.netG_B], False)
        self.backward_G() ## calculate gradients for G_A and G_B
        #print(self.optimizer_G.optimizer)
        #print(self.optimizer_G.optimizer._parameter_list)
        ###print("=======================  network G =========================")
        ###for para in self.optimizer_G.optimizer._parameter_list:
        ###    print("======================= param ========================")
        ###    print(para.name, para.numpy())
        ###    print("======================param gradient===================")
        ###    print(para.name, para.gradient())
        self.optimizer_G.optimizer.minimize(self.loss_G)
        #print(self.netG_A.state_dict())
        ###print("=============================================================")
        self.optimizer_G.optimizer.clear_gradients()
        #self.netG_A.clear_gradients()
        #self.netG_B.clear_gradients()

        #self.set_stop_gradient([self.netG_A, self.netG_B], True)
        self.set_stop_gradient([self.netD_A], False)
        self.backward_D_A() ### calculate gradients for D_A
        ###print("=======================  network DA =========================")
        ###for para in self.optimizer_D_A.optimizer._parameter_list:
        ###    print("======================= param ========================")
        ###    print(para.name, para.numpy())
        ###    print("======================param gradient===================")
        ###    print(para.name, para.gradient())
        self.optimizer_D_A.optimizer.minimize(self.loss_D_A)
        ###print("=============================================================")
        self.optimizer_D_A.optimizer.clear_gradients()

        #self.set_stop_gradient([self.netG_A, self.netG_B], True)
        self.set_stop_gradient([self.netD_B], False)
        self.backward_D_B() ### calculate gradients for D_B
        ###print("=======================  network DB =========================")
        ###for para in self.optimizer_D_B.optimizer._parameter_list:
        ###    print("======================= param ========================")
        ###    print(para.name, para.numpy())
        ###    print("======================param gradient===================")
        ###    print(para.name, para.gradient())
        self.optimizer_D_B.optimizer.minimize(self.loss_D_B)
        ###print("=============================================================")
        self.optimizer_D_B.optimizer.clear_gradients()
        #self.loss_D = self.loss_D_A + self.loss_D_B
        #self.optimizer_D.optimizer.minimize(self.loss_D)
        #self.optimizer_D.optimizer.clear_gradients()

    @fluid.dygraph.no_grad
    def test_single_side(self):
        generator = getattr(self, 'netG_%s' % self.direction[0])
        self.fake_B = generator(self.real_A)

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
        lr_dict['optim_D_A'] = self.optimizer_D_A.optimizer.current_step_lr()
        lr_dict['optim_D_B'] = self.optimizer_D_B.optimizer.current_step_lr()
        return lr_dict

    def set_stop_gradient(self, nets, stop_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = stop_grad

    @fluid.dygraph.no_grad
    def test_single_side(self, direction):
        generator = getattr(self, 'netG_%s' % direction[0])
        self.fake_B = generator(self.real_A)

    def evaluate_mode(self, step):
        ret = {}
        self.is_best = False
        save_dir = os.path.join(self.args.save_dir, 'eval', str(step))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.netG_A.eval()
        self.netG_B.eval()
        for direction in ['AtoB', 'BtoA']:
            eval_dataloader = getattr(self, 'eval_dataloader_' + direction)
            fakes, names = [], []
            cnt = 0
            for i, data_i in enumerate(eval_dataloader):
                self.set_single_input(data_i)
                self.test_single_side(direction)
                fakes.append(self.fake_B.detach().numpy())
                for j in range(len(self.fake_B)):
                    save_dir = os.path.join(self.args.save_dir, 'eval_image')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    name = 'fake_' + direction + str(i + j) + '.png'
                    save_path = os.path.join(save_dir, name)
                    fake_im = util.tensor2img(self.fake_B[j])
                    util.save_image(fake_im, save_path)
                    
            suffix = direction[-1]
            fluid.disable_imperative()
            fid = get_fid(fakes, self.inception_model, getattr(self, 'npz_%s' % direction[-1]), self.args.inception_model)
            fluid.enable_imperative() 
            if fid < getattr(self, 'best_fid_%s' % suffix):
                self.is_best = True
                setattr(self, 'best_fid_%s' % suffix, fid)
            fids = getattr(self, 'fids_%s' % suffix)
            fids.append(fid)
            if len(fids) > 3:
                fids.pop(0)
            ret['metric/fid_%s' % suffix] = fid
            ret['metric/fid_%s-mean' % suffix] = sum(getattr(self, 'fids_%s' % suffix)) / len(getattr(self, 'fids_%s' % suffix))
            ret['metric/fid_%s-best' % suffix] = getattr(self, 'best_fid_%s' % suffix)

        self.netG_A.train()
        self.netG_B.train()
        return ret

