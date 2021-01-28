import numpy as np
import matplotlib.pyplot as plt
import torch
from convae import *

def plot_c(spam, title=""):
    x, y = np.meshgrid(np.arange(60), np.arange(60))
    torch_frame_py = spam.T.reshape(4, 2, 60, 60)
    z0 = torch_frame_py[0, 0, x, y]
    z1 = torch_frame_py[1, 0, x, y]
    z2 = torch_frame_py[2, 0, x, y]
    z3 = torch_frame_py[3, 0, x, y]

    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(title)

    # plt.subplot(2, 2, 1)
    pl0 = axs[0, 0].contourf(x, y, z0)
    cb0 = plt.colorbar(pl0, fraction=0.046, pad=0.04, ax=axs[0,0])  #, ticks=v1)

    # plt.subplot(2, 2, 2)
    pl1 = axs[0, 1].contourf(x, y, z1)
    cb1 = plt.colorbar(pl1, fraction=0.046, pad=0.04, ax=axs[0,1])  #, ticks=v1)

    # plt.subplot(2, 2, 3)
    pl2 = axs[1, 0].contourf(x, y, z2)
    cb2 = plt.colorbar(pl2, fraction=0.046, pad=0.04, ax=axs[1,0])  #, ticks=v1)

    # plt.subplot(2, 2, 4)
    pl3 = axs[1, 1].contourf(x, y, z3)
    cb3 = plt.colorbar(pl3, fraction=0.046, pad=0.04, ax=axs[1,1])  #, ticks=v1)

    plt.savefig('./data/' + title.replace(" ", "") + '.png')
    # plt.close()
    plt.show()

# Device configuration
device = torch.device('cuda')
print("device is: ", device)

init = np.load("./Autoencoders/ConvolutionalAe/latent_initial_4.npy")
print(init)
np.save("./Autoencoders/ConvolutionalAe/latent_initial_4_float64.npy", init.astype(np.float64))

inputs = np.load("x.npy")
print(inputs.shape)
# print(jac)

jac = np.load("jacobian.npy")
print(jac.shape)
# print(jac)

res = np.load("res.npy")
print(res.shape)

#plot from ithaca final jac
x, y = np.meshgrid(np.arange(60), np.arange(60))
torch_frame_py = res.T.reshape(1, 2, 60, 60)
z0 = torch_frame_py[0, 0, x, y]

pl0 = plt.contourf(x, y, z0)
cb0 = plt.colorbar(pl0, fraction=0.046, pad=0.04)  #, ticks=v1)
plt.title("residual")
plt.show()

sys = np.load("system_df.npy")
print(sys.shape, sys[:10, :10], sys[0, 59], sys[0, 60], sys[0, 61])
# plt.spy(sys[-100:, -100:])
# plt.show()

torch_grad = np.load("torch_grad.npy")
print(torch_grad.shape)

# central2 = np.load("fjac_central_2.npy")
# print(central2.shape)

# central3 = np.load("fjac_central_3.npy")
# print(central3.shape)

central5 = np.load("fjac_central_5.npy")
print(central5.shape)

# print("mean, max 1e-2", np.mean(np.abs(jac-central2))/np.mean(np.abs(central2)), np.max(np.abs(jac-central2))/np.max(np.abs(jac)))
# print("mean, max 1e-3", np.mean(np.abs(jac-central3))/np.mean(np.abs(central3)), np.max(np.abs(jac-central3))/np.max(np.abs(jac)))
print("mean, max 1e-5", np.mean(np.abs(jac+central5))/np.mean(np.abs(jac)), np.max(np.abs(jac+central5))/np.max(np.abs(jac)))

# print("sys", sys == 0.)
# print("TORCH", torch_grad)
# print("JACOBIAN", jac[:10, :] )

# reproduce in pytorch
inputs_torch = torch.from_numpy(inputs).transpose(0, 1)
print("inputs shape", inputs_torch.shape)
inputs_repeated = inputs_torch.repeat(3600, 1).requires_grad_(True)
print("inputs shape", inputs_repeated.shape)
grad_output = torch.eye(7200).to(device, dtype=torch.float)
print("ATTE", grad_output.type())
decoder = torch.jit.load("./Autoencoders/ConvolutionalAe/decoder_gpu_4.pt")
output = decoder(inputs_repeated.to(device, dtype=torch.float))

jac_torch = torch.zeros([7200, 4])
for i in range(2):
    output.backward(grad_output[i*3600:(i+1)*3600, :], retain_graph=True)
    jac_torch[i*3600:(i+1)*3600, :] = torch.autograd.grad(output, inputs_repeated, grad_output[i*3600:(i+1)*3600, :], retain_graph=True, create_graph=True)[0]
    # jac_torch[i*100:(i+1)*100, :] = inputs_repeated.grad.data
print("pytorch shape", jac_torch.shape)

#plot from ithaca final jac
plot_c(jac, 'Composite Jacobian from OF')

#plot from ithaca final jac
plot_c(central5, 'Composite Jacobian with Central FD from OF')

#plot from torch
plot_c(jac_torch.detach().numpy(), "Decoder Jacobian from pytorch jit scripted")
del inputs_repeated
del output
del grad_output
del jac_torch

#plot from ithaca
plot_c(torch_grad, "Decoder Jacobian from libtorch OF")

DOMAIN_SIZE = 60
DIM = 2  # number of components of velocity field
WM_PROJECT = "../../"
NUM_EPOCHS = 1000
HIDDEN_DIM = 4
BATCH_SIZE = 40
LEARNING_RATE = 1e-4
ORDER = 4
checkpoint_dir = "./Autoencoders/ConvolutionalAe/checkpoint/"
model_dir = "./Autoencoders/ConvolutionalAe/model/"

# Device configuration
device = torch.device('cuda')
print("device is: ", device)

# snapshots have to be clipped before
snap_vec = np.load("./npSnapshots.npy")
assert np.min(snap_vec) >= 0., "Snapshots should be clipped"

# specify how many samples should be used for training and validation
n_total = snap_vec.shape[1]
n_train = n_total-n_total//6
print("Dimension of validation set: ", n_total-n_train)

# scale the snapshots
nor = Normalize(snap_vec, center_fl=True)
snap_framed = nor.framesnap(snap_vec)
snap_scaled = nor.scale(snap_framed)
snaps_torch = torch.from_numpy(snap_scaled)
print("snapshots shape", snap_scaled.shape)
print("Min max after scaling: ", np.min(snap_scaled), np.max(snap_scaled))

# start model
model = AE(
    HIDDEN_DIM,
    scale=(nor.min_sn, nor.max_sn),
    #mean=nor.mean(device),
    domain_size=DOMAIN_SIZE,
    use_cuda=True).to(device)

modello = torch.load("./model_"+str(HIDDEN_DIM)+".ckpt")
model.load_state_dict(modello['state_dict'])

initial = model.encoder(snaps_torch[:1, :, :, :].to(
            device,
            dtype=torch.float))#.detach().to(torch.device('cpu'), dtype=torch.double).numpy()
print("initial latent variable shape : ", initial)

decoded = model.decoder(initial.to(device, dtype=torch.float32))
plot_snapshot(decoded.detach().cpu().numpy().reshape(-1, 2, 60, 60), 0, title="decoded")


rec_initial = model(snaps_torch[:1, :, :, :].to(
            device,
            dtype=torch.float))
print("rec latent variable shape : ", rec_initial.shape)
A = rec_initial.detach().cpu().numpy().reshape(-1, 2, 60, 60)[0, 0, :, :] > 0
plt.spy(A)
plt.show()

plot_snapshot(rec_initial.detach().cpu().numpy().reshape(-1, 2, 60, 60), 0)

# reproduce in pytorch
torch.cuda.empty_cache()
inputs_torch = torch.from_numpy(inputs).transpose(0, 1)
print("inputs shape", inputs_torch.shape)
inputs_repeated = inputs_torch.repeat(3600, 1).requires_grad_(True)
print("inputs shape", inputs_repeated.shape)
grad_output = torch.eye(7200).to(device, dtype=torch.float)
output = model.decoder(inputs_repeated.to(device, dtype=torch.float))
plot_snapshot(output.detach().cpu().numpy().reshape(-1, 2, 60, 60), 0)


jac_torch_t = torch.zeros([7200, 4])
for i in range(2):
    output.backward(grad_output[i*3600:(i+1)*3600, :], retain_graph=True)
    jac_torch_t[i*3600:(i+1)*3600, :] = torch.autograd.grad(output, inputs_repeated, grad_output[i*3600:(i+1)*3600, :], retain_graph=True, create_graph=True)[0]
    # jac_torch_t[i*100:(i+1)*100, :] = inputs_repeated.grad.data
print("pytorch shape", jac_torch_t.shape)

# output.backward(grad_output)
# jac_torch_t = inputs_repeated.grad.data
print("pytorch shape", jac_torch_t.shape)

#plot from ithaca
plot_c(jac_torch_t.detach().numpy(), "Decoder Jacobian from pytorch not jit scripted")

print("TORCH", torch_grad[:10, 0], jac_torch_t[:10, 0], jac_torch_t[:10, 0])
