import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from .base_resnet_distiller import BaseResnetDistiller
from utils import util
from utils.weight_transfer import load_pretrained_weight
from metric import compute_fid
from model import loss

class ResnetDistiller(BaseResnetDistiller):
    def __init__(self, args):
        assert args.isTrain
        super(ResnetDistiller, self).__init__(args)
        self.best_fid = 1e9
        self.fids = []
        self.npz = np.load(args.real_stat_path)

    def forward(self):
        self.Tfake_B = self.netG_teacher(self.real_A)
        self.netG_teacher.eval()
        self.Tfake_B.stop_gradient = True
        self.Sfake_B = self.netG_student(self.real_A)

    def calc_distill_loss(self):
        losses = []
        for i, netA in enumerate(self.netAs):
            assert isinstance(netA, Conv2D)
            n = self.mapping_layers[i]
            Tact = self.Tacts[n]
            Sact = self.Sacts[n]
            ### 1x1 conv to match channels
            Sact = netA(Sact)
            loss = fluid.layers.mse_loss(Sact, Tact)
            setattr(self, 'loss_G_distill%d' % i, loss)
            losses.append(loss)
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

    def set_stop_gradient(self, nets, stop_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.stop_gradient = stop_grad

    def optimize_parameter(self):
        self.forward()

        self.set_stop_gradient(self.netD, False)
        self.backward_D()
        self.optimizer_D.optimizer.minimize(self.loss_D)
        self.optimizer_D.optimizer.clear_gradients()

        self.set_stop_gradient(self.netD, True)
        self.backward_G()
        self.optimizer_G.optimizer.minimize(self.loss_G)
        self.optimizer_G.optimizer.clear_gradients()

    def load_networks(self):
        if self.args.restore_pretrained_G_path is not None:
            util.load_network(self.netG_pretrained, self.args.restore_pretrained_G_path)
            load_pretrained_weight(self.args.pretrained_netG, self.args.student_netG, self.netG_pretrained, self.netG_student, self.args.pretrained_ngf, self.args.student_ngf)
            import pickle
            pickle.dump(self.net_student.state_dict(), open('student.pkl', 'wb'))
            sys.exit(0)
            del self.netG_pretrained
        super(ResnetDistiller, self).load_networks()

    def evaluate_model(self, step):
        self.is_best = False
        save_dir = os.path.join(self.args.save_dir, 'eval', str(step))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.netG_student.eval()
        cnt = 0
        for i, data_i in enumerate(self.eval_dataloader):
            self.set_single_input(data_i)
            self.test()
            fakes.append(self.Sfake_B.detach().numpy())
            for j in range(len(self.Sfake_B)):
                if cnt < 10:
                    Sname = 'Sfake_' + str(i + j) + '.png'
                    Tname = 'Tfake_' + str(i + j) + '.png'
                    Sfake_im = util.tensor2img(self.Sfake_B[j])
                    Tfake_im = util.tensor2img(self.Tfake_B[j])
                    util.save_image(Sfake_im, os.path.join(save_dir, Sname))
                    util.save_image(Tfake_im, os.path.join(save_dir, Tname))
                cnt += 1
                
        suffix = direction[-1]
        fluid.disable_imperative()
        fid = get_fid(fakes, self.inception_model, self.npz, self.args.inception_model)
        fluid.enable_imperative() 
        if fid < self.best_fid:
            self.is_best = True
            self.best_fid = fid
        print("fid score is: %f, best fid score is %f" % (fid, self.best_fid))
        self.fids.append(fid)
        if len(self.fids) > 3:
            self.fids.pop(0)
        ret['metric/fid'] = fid
        ret['metric/fid-mean'] = sum(self.fids) / len(self.fids)
        ret['metric/fid-best'] = self.best_fid

        self.netG_student.train()
        return ret

