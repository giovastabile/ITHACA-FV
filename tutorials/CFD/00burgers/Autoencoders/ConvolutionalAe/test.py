import numpy as np
import torch
import torch.nn as nn
import clock
import convae
import time

print("torch version: ", torch.__version__)
loaded = np.load("latent_initial_4.npy")
print(loaded)

@clock.clock
def whole_jac(dev, inputs, output_grad):
    device = torch.device(dev)
    use_cuda = False if dev=='cpu' else True
    model = convae.AE(4, scale=(min_sn,max_sn), domain_size=60, use_cuda=use_cuda)
    model.load_state_dict(torch.load('model.ckpt', map_location=device))
    # model.eval()
    inputs = inputs.to(device)
    inputs.requires_grad_(True)
    outputs = model.decoder(inputs)
    outputs.backward(output_grad.to(device))
    # print(inputs.grad.data.size())

def jacobian_piece_by_piece():
    device = torch.device('cuda')
    model = convae.AE(4, scale=(min_sn,max_sn), domain_size=60, use_cuda=True)
    model.load_state_dict(torch.load('model.ckpt', map_location=device))
    # model.eval()
    inputs = torch.ones((1, 4)).to(device)
    inputs.requires_grad_(True)
    outputs = model.decoder(inputs)
    outputs.backward((torch.eye(10800)[:1, :]).to(device))
    print(inputs.grad.data.size())

def test_jac():
    inputs = torch.ones((2, 2))
    output_grad = torch.Tensor([[1, 0, 0], [0, 1, 0]])
    test = Test()
    inputs.requires_grad_(True)
    outputs = test(inputs)
    outputs.backward(output_grad)
    print(inputs.grad.data)

class Test(nn.Module):
    def __init__(self, A=None, use_cuda=False):
        super(Test, self).__init__()
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            # self.encoder.cuda()
            # self .decoder.cuda()
        if A:
            self.A = A
        else:
            self.A = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)

    def forward(self, x):
        return nn.functional.linear(x, self.A)


test_jac()

snapshots = np.load("../../npSnapshots.npy")
max_sn = np.max(snapshots)
min_sn = np.min(snapshots)

print("Singel component CPU")
whole_jac('cpu', torch.ones((1, 4)), torch.ones(1, 10800))

print("Whole Jacobian CPU")
whole_jac('cpu', torch.ones((7200, 4)), torch.eye(10800)[:7200, :])

print("Whole Jacobian GPU")
whole_jac('cuda', torch.ones((7200, 4)), torch.eye(10800)[:7200, :])

# device = torch.device('cuda')
# model = convae.AE(4, scale=max_sn, domain_size=60, use_cuda=True)
# model.load_state_dict(torch.load('model.ckpt', map_location=device))

# start = time.time()
# torch.autograd.functional.jacobian(model.decoder, torch.ones((1, 4)).to(device))
# print("time: ", time.time()-start)
