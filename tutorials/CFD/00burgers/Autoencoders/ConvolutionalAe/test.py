import numpy as np
import torch
import clock
import convae
import time

print("torch version: ", torch.__version__)
loaded = np.load("latent_initial.npy")
print(loaded)
@clock.clock
def whole_jac(dev, inputs, output_grad):
    device = torch.device(dev)
    use_cuda = False if dev=='cpu' else True
    model = convae.AE(4, scale=max_sn, domain_size=60, use_cuda=use_cuda)
    model.load_state_dict(torch.load('model.ckpt', map_location=device))
    # model.eval()
    inputs = inputs.to(device)
    inputs.requires_grad_(True)
    outputs = model.decoder(inputs)
    outputs.backward(output_grad.to(device))
    # print(inputs.grad.data.size())

def jacobian_piece_by_piece():
    device = torch.device('cuda')
    model = convae.AE(4, scale=max_sn, domain_size=60, use_cuda=True)
    model.load_state_dict(torch.load('model.ckpt', map_location=device))
    # model.eval()
    inputs = torch.ones((1, 4)).to(device)
    inputs.requires_grad_(True)
    outputs = model.decoder(inputs)
    outputs.backward((torch.eye(10800)[:1, :]).to(device))
    print(inputs.grad.data.size())

snapshots = np.load("../../npSnapshots.npy")
max_sn = np.max(snapshots)

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
