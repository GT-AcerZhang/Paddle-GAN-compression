import os
import numpy as np
import paddle.fluid as fluid
from distiller.base_resnet_distiller import BaseResnetDistiller
from model.super_modules import SuperConv2D
from configs.resnet_configs import get_configs
from model import loss

class ResnetSupernet(BaseResnetDistiller):
    def __init__(self, args):
        assert args.isTrain
        assert 'super' in args.student_netG
        super(ResnetSupernet, self).__init__(args)
        self.best_fid_largest = 1e9
        self.best_fid_smallest = -1e9
        self.fids_largest, self.fids_smallest = [], []
        if args.config_set is not None:
            assert args.config_str is None
            self.configs = get_configs(args.config_set)
            self.args.eval_mode = 'both'
        else:
            assert args.config_str is not None
            self.configs = SingleConfigs(decode_config(args.config_str))
            self.opt.eval_mode = 'largest'

    def forward(self, config):
        self.Tfake_B = self.netG_teacher(self.real_A)
        self.Tfake_B.stop_gradient = True
        self.netG_student.configs = config
        self.Sfake_B = self.netG_student(self.real_A)
        #print("================================   TEACHER   ===========================")
        #for n, sublayer in self.netG_teacher.named_sublayers():
        #    print(n, sublayer)
        #print("========================================================================")
        #print("================================   STUDENT   ===========================")
        #for n, sublayer in self.netG_student.named_sublayers():
        #    print(n, sublayer)
        #print("========================================================================")
        #print(self.Tfake_B.shape, self.Sfake_B.shape)

    def calc_distill_loss(self):
        losses = []
        print(self.Tacts)
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, SuperConv2D)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            Sact = netA(Sact, {'channel': netA._num_filters})
            loss = fluid.layers.mse_loss(Sact, Tact)
            setattr(self, 'loss_G_distill%d' % i, loss)
            losses.append(loss)
        sys.exit(0)
        return sum(losses)

    def backward_G(self):
        self.loss_G_recon = loss.recon_loss(self.args.recon_loss_mode, self.Sfake_B, self.Tfake_B) * self.args.lambda_recon
        fake = self.Sfake_B
        pred_fake = self.netD(fake)
        self.loss_G_gan = loss.gan_loss(self.args.gan_loss_mode, pred_fake, True, for_discriminator=False) * self.args.lambda_gan
        if self.args.lambda_distill > 0:
            self.loss_G_distill = self.calc_distill_loss() * self.args.lambda_distill
        else:
            self.loss_G_distill = 0
        self.loss_G = self.loss_G_gan + self.loss_G_recon + self.loss_G_distill
        self.loss_G.backward()

    def optimize_parameter(self):
        config = self.configs.sample()
        self.forward(config=config)
        self.backward_D()
        self.optimizer_D.minimize(self.loss_D)
        self.optimizer_D.clear_gradients()

        self.backward_G()
        self.optimizer_G.minimize(self.loss_G)
        self.optimizer_G.clear_gradients()

    def evaluate_model(self, step):
        pass
