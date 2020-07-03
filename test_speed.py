import numpy as np
import time
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import BatchNorm, InstanceNorm
from paddle.fluid.dygraph.base import to_variable

from models.network import define_G

fake_img = np.random.random((1, 3, 256, 256)).astype('float32')
###16_16_32_16_32_32_16_16
#configs = {'channels': [16,16,32,16,32,32,16,16]}
configs = 64
#place = fluid.CPUPlace()
place = fluid.CUDAPlace(0)
with fluid.dygraph.guard(place=place):
    fake_inp = to_variable(fake_img) 
    
#    netG = 'sub_mobile_resnet_9blocks'
    netG = 'resnet_9blocks'
    
    model = define_G(3, 3, configs, netG, 'instance', 0, 9)
    
    for idx in range(100):
        output = model(fake_inp)
    
    stime = time.time()
    for idx in range(100):
        output = model(fake_inp)
    etime = time.time()
    
    avg_time = (etime - stime)/100
    print(str(avg_time) + 's')
