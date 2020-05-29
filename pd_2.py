import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2DTranspose

class MyLayer(fluid.Layer):
    def __init__(self):
        super(MyLayer, self).__init__()
        self.model = fluid.dygraph.LayerList([])
        self.model.append(Conv2DTranspose(3, 5, filter_size=3, stride=2, padding=1))

    def forward(self, x):
        for i, sublayer in enumerate(self.model):
            x = sublayer(x)
        return x

with fluid.dygraph.guard():
    model = MyLayer()
    optimizer_D = fluid.optimizer.Adam(learning_rate=2e-4, beta1=0.5, beta2=0.999, parameter_list=model.parameters())
    state_dict, _ = fluid.load_dygraph('./conv2d_transpose')
    model.set_dict(state_dict)

    print(model)

    import pickle
    import numpy as np
    from paddle.fluid.dygraph.base import to_variable

    for i in range(3):
        data = pickle.load(open('data_.pkl', 'rb'))
        data_A = to_variable(np.array(data['A']))

        output = model(data_A)
        ones = fluid.layers.fill_constant(shape = fluid.layers.shape(output), value=0.0, dtype='float32')
        loss = fluid.layers.reduce_mean(fluid.layers.mse_loss(output, ones))
        print(np.sum(np.abs(loss.numpy())))
        loss.backward()
        optimizer_D.minimize(loss)
        optimizer_D.clear_gradients()
    #fluid.save_dygraph(model.state_dict(), './conv2d_transpose')
    
