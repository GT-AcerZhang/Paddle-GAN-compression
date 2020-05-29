import paddle.fluid as fluid
from data_reader import data_reader

def create_data(args):
    reader = data_reader(args)
    A_reader, B_reader, _, _, batch_num, a_id2name, b_id2name = reader.make_data()
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


def create_eval_data(args, direction='AtoB'):
    reader = data_reader(args, run_test=True)
    _, _, A_reader_test, B_reader_test, batch_num, a_id2name, b_id2name = reader.make_data()

    loader = fluid.io.DataLoader.from_generator(
        capacity=4,
        iterable=True,
        use_double_buffer=True)

    if direction == 'AtoB':
        reader = A_reader_test
    else:
        reader = B_reader_test

    loader.set_batch_generator(
        reader,
        places=fluid.CUDAPlace(0) 
        if args.use_gpu else fluid.cpu_places()) ### fluid.cuda_places()
    return loader
