import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
import argparse
from convae import *
import clock
domain_size = 60

def main(args):
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.latent_dim
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    snapshots = np.load("../../npSnapshots.npy")
    n_train = snapshots.shape[1]

    print("use cuda is: ", args.device)
    # reshape as (train_samples, channel, y, x)
    snapshots = snapshots.T
    snapshots = np.vstack((snapshots, snapshots[150:200, :], snapshots[350:400, :], snapshots[550:600, :], snapshots[750:800, :]))
    snapshots = snapshots.reshape((-1, 3, domain_size, domain_size))
    max_sn = np.max(np.abs(snapshots))
    snapshots /= max_sn
    snapshots = torch.from_numpy(snapshots)
    print("snapshots shape: ", snapshots.size(), snapshots[0, :, :, :].size())

    # Device configuration
    device = torch.device('cuda:0' if args.device else 'cpu')
    # device = 'cpu'
    print("device is: ", device)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=snapshots,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    model = AE(HIDDEN_DIM, scale=max_sn, domain_size=domain_size, use_cuda=args.device).to(device)

    # Loss and optimizer
    criterion = nn.L1Loss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in model.parameters():
    #     reg_loss += l1_crit(param)

    # factor = 0.005
    # criterion += factor * reg_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_list = []

    # Train the model
    total_step = len(train_loader)
    for epoch in range(1, NUM_EPOCHS):
        for snapshot in train_loader:

            snapshot = snapshot.to(device, dtype=torch.float)
            # Forward pass
            outputs = model(snapshot)
            loss = criterion(outputs.reshape((-1, 3, domain_size, domain_size)), snapshot)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss)

        # plt.ion()
        if epoch % args.iter == 0:
            plot_snapshot(max_sn*outputs.detach().cpu().numpy().reshape((-1, 3, domain_size, domain_size)), 0)
            plot_snapshot(max_sn*snapshot.detach().cpu().numpy().reshape((-1, 3, domain_size, domain_size)), 0)
            # plt.show()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                .format(epoch, NUM_EPOCHS, epoch, total_step, loss.item()))
            plt.plot(range(epoch//args.iter), np.log10(loss_list[::args.iter]))
            # plt.savefig('./train_cae.png')
            # plt.draw()
            # plt.pause(0.05)
            plt.show()

    plt.plot(range(1, NUM_EPOCHS), np.log10(loss_list))
    plt.show()

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
    summary(model, input_size=(3, domain_size, domain_size))

    # Save the initial value of the latent variable
    initial = model.encoder(snapshots[:1, :, :, :].to(device, dtype=torch.float)).detach().cpu().numpy()
    print("initial latent variable shape : ", initial)
    np.save("latent_initial.npy", initial)

    # Save decoder
    device = 'cpu'
    model.decoder.to(device)
    sm = torch.jit.script(model.decoder)
    sm.save('decoder.pt')



if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num_epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning_rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch', '--batch_size', default=20, type=int, help='learning rate')
    parser.add_argument('-dim', '--latent_dim', default=4, type=int, help='learning rate')
    parser.add_argument('-device', '--device', default=1, type=int, help='whether to use cuda')
    parser.add_argument('-i', '--iter', default=2, type=int, help='epoch when visualization runs')
    args = parser.parse_args()

    main(args)
