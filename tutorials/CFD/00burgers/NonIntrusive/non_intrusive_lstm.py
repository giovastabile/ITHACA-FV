import torch
import numpy as np
from torchsummary import summary
import torch.nn as nn

import matplotlib.pyplot as plt
import argparse
from lstm import *

def main(args):
    # load training inputs
    array = []
    with open('../ITHACAoutput/Offline/Training/mu_samples_mat.txt') as f:
        for i, line in enumerate(f):
            array.append([*line.split()])

    array_ = [[float(elem) for elem in item] for item in array]
    x = np.array(array_)
    print("inputs shape: ", x.shape)

    # get the number of training params apart from time
    train_params = np.load("../parTrain.npy").reshape(-1)
    n_train_params = train_params.shape[0]

    # loading training outputs
    output_pre = np.load(
        '../ITHACAoutput/red_coeff/red_coeff_mat.npy').squeeze()
    n_time_samples_times_n_param = output_pre.shape[0]
    n_time_samples = n_time_samples_times_n_param // n_train_params
    print("number of time samples: ", n_time_samples)

    time_samples = output_pre[:n_time_samples, 0]
    output = output_pre[:, 1:]
    print("outputs shape: ", output.shape, np.max(output), np.min(output))
    print("time samples: ", time_samples)

    # LSTM

    # Device configuration
    device = torch.device('cuda' if eval(args.device) else 'cpu')
    print("device is: ", device)

    input_dim = x.shape[1]
    hidden_dim = output.shape[1]
    n_layers = 2
    n_train = 10000
    model = ReducedCoeffsTimeSeries().to(device)

    # dataloader
    x = torch.from_numpy(x.reshape(n_train_params, n_time_samples, x.shape[1]))
    output = torch.from_numpy(output.reshape(n_train_params, n_time_samples, output.shape[1]))

    val_input = x[4, :, :].unsqueeze(0)
    x = torch.cat([x[:4, :, :], x[5:, :, :]])
    val_output  = output[4, :, :].unsqueeze(0)
    output = torch.cat([output[:4, :, :], output[5:, :, :]])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [9000])
    loss_list = []
    val_list = []

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
            if val_error < 0.16:
                optimizer.param_groups[0]['lr'] = args.learning_rate * 0.1
            print('Epoch [{}/{}], Train loss: {:.12f}, Validation loss: {:.12f}'.format(epoch, args.num_epochs, loss.item(), val_error))

            if val_error < 0.01:
                break

    plt.subplot(2, 1, 1)
    plt.plot(range(len(loss_list)), np.log10(loss_list))
    plt.ylabel('training error')

    plt.subplot(2, 1, 2)
    plt.plot(range(len(val_list)), np.log10(val_list))
    plt.xlabel('epochs')
    plt.ylabel('validation error')

    plt.show()

    torch.save(model.state_dict(), 'model.ckpt')

    model(inputs)[0]


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n',
                        '--num_epochs',
                        default=5000,
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
                        default=50,
                        type=int,
                        help='epoch when visualization runs')
    args = parser.parse_args()

    main(args)
