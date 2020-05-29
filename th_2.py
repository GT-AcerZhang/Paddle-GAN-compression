import torch
import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(3, 5, kernel_size=3, stride=2, padding=1))

    def forward(self, input):
        #out = self.model(input)
        for name, module in self.model.named_modules():
            print(name)
        return self.model(input)

model = Net()
model = model.cuda()

import pickle
import numpy as np

weights = torch.load('./conv2d_transpose_th')
model.load_state_dict(weights)
for i in range(3):
    data = pickle.load(open('data_.pkl', 'rb'))
    data_A = torch.from_numpy(data['A'].numpy()).cuda()
    out = model(data_A)
    ones = torch.zeros(out.shape).cuda()
    loss = torch.nn.functional.mse_loss(out, ones, reduction='mean')
    print(loss)

    optim = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim.zero_grad()
    loss.backward()
    optim.step()
#torch.save(model.cpu().state_dict(), './conv2d_transpose_th')


#data_np = np.random.random((64, 32, 3, 3))
#print(data_np)
#data = torch.from_numpy(data_np).cuda()
#out = model(data)
#print(out.numpy())
