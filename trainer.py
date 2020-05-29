import os
import numpy as np
import paddle.fluid as fluid


class Trainer:
    def __init__(self, task):
        if task == 'train':
            from arguments.train_arguments import TrainArgs as args
            from models import create_model as create_model
        elif task == 'distill':
            from arguments.distill_arguments import DistillArgs as args
            from distiller import create_distiller as create_model
        else:
            pass
