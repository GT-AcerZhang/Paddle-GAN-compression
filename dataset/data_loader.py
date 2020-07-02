import paddle.fluid as fluid
from data_reader import data_reader

def create_data(cfgs, direction='AtoB', eval_mode=False):
    if eval_mode == False:
        mode = 'TRAIN'
    else:
        mode = 'EVAL'
    reader = data_reader(cfgs, mode=mode)
    dreader, id2name = reader.make_data(direction)

    if cfgs.use_parallel:
        dreader = fluid.contrib.reader.distributed_batch_reader(dreader)

    #### id2name has something wrong when use_multiprocess
    loader = fluid.io.DataLoader.from_generator(
        capacity=4,
        return_list=True,
        use_multiprocess=cfgs.use_multiprocess)

    loader.set_batch_generator(
        dreader,
        places=cfgs.place) 
    return loader, id2name
 
def create_eval_data(cfgs, direction='AtoB'):
    return create_data(cfgs, eval_mode=True)
