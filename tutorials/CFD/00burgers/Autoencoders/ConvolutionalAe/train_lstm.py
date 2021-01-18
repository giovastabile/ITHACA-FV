import torch
import numpy as np
from torchsummary import summary
import torch.nn as nn

import matplotlib.pyplot as plt
import argparse
from lstm import *
from convae import *

def main(args):
    WM_PROJECT = "../../"
    HIDDEN_DIM = args.latent_dim
    DOMAIN_SIZE = 60
    DIM = 2

    # Device configuration
    device = torch.device('cpu')
    print("device is: ", device)

    # snapshots have to be clipped before
    snap_vec = np.load(WM_PROJECT + "npSnapshots.npy")
    assert np.min(snap_vec) >= 0., "Snapshots should be clipped"

    n_total = snap_vec.shape[1]
    n_train = n_total-n_total//6

    # scale the snapshots
    nor = Normalize(snap_vec, center_fl=True)
    snap_framed = nor.framesnap(snap_vec)
    snap_scaled = nor.scale(snap_framed)
    snap_torch = torch.from_numpy(snap_scaled).to("cpu", dtype=torch.float)
    print("snapshots shape", snap_scaled.shape)
    print("Min max after scaling: ", np.min(snap_scaled), np.max(snap_scaled))

    # load autoencoder
    model = AE(
        HIDDEN_DIM,
        scale=(nor.min_sn, nor.max_sn),
        #mean=nor.mean(device),
        domain_size=DOMAIN_SIZE,
        use_cuda=args.device).to(device)

    modello = torch.load("./model_"+str(args.latent_dim)+".ckpt")
    model.load_state_dict(modello['state_dict'])

    # model.load_state_dict(torch.load("./model_"+str(args.latent_dim)+".ckpt"))
    # model.eval()

    # # plot initial
    # inputs = torch.from_numpy(np.load("latent_initial.npy")).to(device, dtype=torch.float)
    # output = model.decoder.forward(inputs)
    # print("shapeoutput", output)
    # plot_snapshot(output.detach().cpu().numpy().reshape(1, 2, 60, 60), 0, idx_coord=1)

    # reconstruct snapshots
    snap_rec = model(snap_torch).cpu().detach().numpy()
    print("non linear reduction training coeffs: ", snap_rec.shape)
    # plot_compare(snap_framed, nor.frame2d(snap_rec), n_train)

    # evaluate hidden variables
    nl_red_coeff = model.encoder.forward(snap_torch)
    print("non linear reduction training coeffs: ", nl_red_coeff.size())
    nl_red_coeff = nl_red_coeff.cpu().detach().numpy()
    print("max, min : ", np.max(nl_red_coeff), np.min(nl_red_coeff))

    # test max error
    err = nor.frame2d(snap_rec)-snap_framed
    # plot_snapshot(err, 11000, idx_coord=1)
    print(snap_framed.shape, snap_rec.shape)
    err_max = np.max(np.abs(nor.vectorize2d(err)), axis=1)
    norm_max = np.max(np.abs(nor.vectorize2d(snap_framed)), axis=1)
    error_max = err_max/norm_max
    print("error max: ", np.max(error_max), np.min(error_max), np.max(err_max), np.min(err_max) )

    ##################################### TRAIN LSTM
    # load training inputs
    array=[]
    with open(WM_PROJECT+"ITHACAoutput/Offline/Training/mu_samples_mat.txt") as f:
        for i, line in enumerate(f):
            array.append([*line.split()])

    array_ = [[float(elem) for elem in item] for item in array]
    x = np.array(array_)
    print("inputs shape: ", x.shape)

    train_params = np.load(WM_PROJECT+"parTrain.npy").reshape(-1)
    n_train_params = train_params.shape[0]

    # loading training outputs
    output_pre = np.load(WM_PROJECT+'ITHACAoutput/red_coeff/red_coeff_mat.npy').squeeze()
    n_time_samples_times_n_param = output_pre.shape[0]
    n_time_samples = n_time_samples_times_n_param//n_train_params
    print("number of time samples: ", n_time_samples)

    time_samples = output_pre[:n_time_samples, 0]
    output = output_pre[:, 1:]
    print("outputs shape: ", output.shape)
    print("time samples: ", time_samples.shape)

    # Device configuration
    device = torch.device('cuda' if eval(args.device) else 'cpu')
    print("device is: ", device)

    # LSTM
    input_dim = x.shape[1]
    hidden_dim = output.shape[1]
    n_layers = 2
    n_train = 10000
    model = ReducedCoeffsTimeSeries().to(device)

    # dataloader
    x = torch.from_numpy(x.reshape(n_train_params, n_time_samples, x.shape[1]))
    output = torch.from_numpy(nl_red_coeff.reshape(n_train_params, n_time_samples, args.latent_dim))

    val_input = x[4, :, :].unsqueeze(0)
    x = torch.cat([x[:4, :, :], x[5:, :, :]])
    val_output  = output[4, :, :].unsqueeze(0)
    output = torch.cat([output[:4, :, :], output[5:, :, :]])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 5000])
    loss_list = []
    val_list = []
    it = 0
    best = 1.
    for epoch in range(1, args.num_epochs):
        inputs = x[:].to(device, dtype=torch.float)
        forwarded = model(inputs)
        loss = criterion(forwarded.reshape(-1),
                        output.reshape(-1).to(device, dtype=torch.float))
        loss_list.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # plt.ion()
        if epoch % args.iter == 0:
            val_forwarded = model(val_input[:].to(device, dtype=torch.float))
            val_error = np.max(np.abs((val_forwarded.reshape(-1).detach().cpu().numpy
            ()-val_output.reshape(-1).detach().cpu().numpy())))
            val_list.append(val_error)
            print('Epoch [{}/{}], Train loss: {:.12f}, Validation loss: {:.12f}'.format(epoch, args.num_epochs, loss.item(), val_error))

            if val_error < 20:
                optimizer.param_groups[0]['lr'] = 0.0001

            if val_error < 5:
                break

    plt.subplot(2, 1, 1)
    plt.plot(range(len(loss_list)), np.log10(loss_list))
    plt.ylabel('training error')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(val_list)), np.log10(val_list))
    plt.xlabel('epochs')
    plt.ylabel('validation error')

    plt.show()

    torch.save(model.state_dict(), 'lstm_'+str(HIDDEN_DIM)+'.ckpt')
    summary(model, input_size=(1, n_time_samples, 2))
    model(inputs)[0]


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n',
                        '--num_epochs',
                        default=10000,
                        type=int,
                        help='number of training epochs')
    parser.add_argument('-lr',
                        '--learning_rate',
                        default=1.0e-2,
                        type=float,
                        help='learning rate')
    parser.add_argument('-batch',
                        '--batch_size',
                        default=20,
                        type=int,
                        help='batch')
    parser.add_argument('-dim',
                        '--latent_dim',
                        default=4,
                        type=int,
                        help='latent dim')
    parser.add_argument('-device',
                        '--device',
                        default='True',
                        type=str,
                        help='whether to use cuda')
    parser.add_argument('-i',
                        '--iter',
                        default=5,
                        type=int,
                        help='epoch when visualization runs')
    args = parser.parse_args()

    main(args)