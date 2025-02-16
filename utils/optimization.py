import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

#step_per_epoch = 1334

class LinearDecay(LearningRateDecay):
    def __init__(self, learning_rate, step_per_epoch, nepochs, nepochs_decay):
        super(LinearDecay, self).__init__()
        self.learning_rate = learning_rate
        self.nepochs = nepochs
        self.nepochs_decay = nepochs_decay
        self.step_per_epoch = step_per_epoch

    def step(self):
        cur_epoch = np.floor(self.step_num / self.step_per_epoch)
        lr_l = 1.0 - max(0, cur_epoch + 1 - self.nepochs) / float(self.nepochs_decay + 1)
        return self.create_lr_var(lr_l * self.learning_rate)

class Optimizer:
    def __init__(self, lr, scheduler, step_per_epoch, nepochs, nepochs_decay, args, parameter_list=None):
        self.lr = lr
        self.scheduler = scheduler
        self.step_per_epoch = step_per_epoch
        self.nepochs = nepochs
        self.nepochs_decay = nepochs_decay
        self.args = args
        self.parameter_list = parameter_list
        self.optimizer = self.lr_scheduler()

    ### NOTE(ceci3): add more scheduler
    def lr_scheduler(self):
        if self.scheduler == 'linear':
            self.scheduler_lr = LinearDecay(self.lr, self.step_per_epoch, self.nepochs, self.nepochs_decay)
        elif self.scheduler == 'step':
            pass
        elif self.scheduler == 'cosine':
            pass
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)

        optimizer = fluid.optimizer.Adam(learning_rate=self.scheduler_lr, beta1=self.args.beta1, beta2=0.999, parameter_list=self.parameter_list)
        return optimizer
