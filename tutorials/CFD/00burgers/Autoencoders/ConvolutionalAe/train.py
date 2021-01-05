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
    DIM = 2
    ORDER = 2

    snapshots = np.load("../../npSnapshots.npy")
    n_total = snapshots.shape[1]
    n_train = 10000
    print("use cuda is: ", args.device)
    # reshape as (train_samples, channel, y, x)
    snapshots = snapshots.T
    # snapshots = np.vstack((snapshots, snapshots[150:200, :], snapshots[350:400, :], snapshots[550:600, :], snapshots[750:800, :]))
    snapshots = snapshots.reshape((-1, 3, domain_size, domain_size))[:, :DIM, :, :]
    print("max, min ", np.max(snapshots), np.min(snapshots))
    max_sn = np.max(snapshots)
    min_sn = np.min(snapshots)
    snapshots -= min_sn
    snapshots /= max_sn-min_sn
    print("max, min ", np.max(snapshots), np.min(snapshots))
    snapshots = torch.from_numpy(snapshots)
    print("snapshots shape: ", snapshots.size(), snapshots[0, :, :, :].size())

    # Device configuration
    device = torch.device('cuda' if eval(args.device) else 'cpu')
    # device = 'cpu'
    print("device is: ", device)

    # Data loader
    train_snap, validation_snap = torch.utils.data.random_split(snapshots, [n_train, n_total-n_train])
    train_loader = torch.utils.data.DataLoader(dataset=train_snap,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

    validation = validation_snap[:].to("cuda", dtype=torch.float)
    validation = validation*(max_sn-min_sn)+min_sn

    model = AE(HIDDEN_DIM, scale=(min_sn,max_sn), domain_size=domain_size, use_cuda=args.device).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in model.parameters():
    #     reg_loss += l1_crit(param)

    # factor = 0.005
    # criterion += factor * reg_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [500, 1500])

    loss_list = []
    val_list = []
    loss = 1
    changed_to_SGD = True
    warming = True

    # Train the model
    total_step = len(train_loader)
    start = time.time()
    for epoch in range(1, NUM_EPOCHS):
        for i, snapshot in enumerate(train_loader):

            snapshot = snapshot.to(device, dtype=torch.float)
            # Forward pass
            outputs = model(snapshot)
            loss = criterion(outputs.reshape((-1, DIM, domain_size, domain_size)), snapshot*(max_sn-min_sn)+min_sn)
            # loss = torch.norm(outputs.reshape((-1, DIM, domain_size, domain_size))-snapshot, p=ORDER)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_list.append(loss.detach().cpu().numpy())

        # validation
        outputs_val = model(validation)
        diff = torch.abs(outputs_val.reshape((-1, 2, domain_size, domain_size))-validation)
        index = torch.argmax(diff.reshape(-1, 2*domain_size**2), dim=1)
        loss_val = torch.max(diff)
        val_list.append(loss_val.detach().cpu().numpy())

        if epoch > 800 and loss_val<0.004:
            break
        # plt.ion()
        if epoch % args.iter == 0:

            print ('Epoch [{}/{}], Time: {} s, Loss: {:.10f}\n Validation, Loss: {:.6f}'.format(epoch, NUM_EPOCHS, time.time()-start, loss.item(), loss_val))
            start = time.time()
            # # validation plot
            # index_list = torch.sort(index, descending=True)[1]
            # plot_two(outputs_val.detach().cpu().numpy().reshape(-1, DIM, domain_size, domain_size), validation.detach().cpu().numpy(), (index_list[:3]).detach().cpu().numpy(), epoch, title="validation")

            # # loss plot
            # loss_plot = torch.norm(outputs-snapshot.reshape(-1, 2*domain_size**2), p=ORDER, dim=0).reshape(1, 2, domain_size, domain_size).detach().cpu().numpy()
            # loss_plot_ = (torch.max(torch.abs(outputs-snapshot.reshape(-1, 2*domain_size**2)), dim=0)[0]).reshape(1, 2, domain_size, domain_size).detach().cpu().numpy()
            # plot_snapshot(loss_plot, 0)
            # plot_two(loss_plot, loss_plot_, [0], epoch, title="loss")

            # reconstruction plot
            # plt.show()
            # # plot_snapshot(max_sn*outputs.detach().cpu().numpy().reshape((-1, DIM, domain_size, domain_size)), 0)
            # plot_snapshot(max_sn*snapshot.detach().cpu().numpy().reshape((-1, DIM, domain_size, domain_size)), 0)
            # plt.show()

            # error plot
            # plt.plot(range(epoch//args.iter), np.log10(loss_list[::args.iter]))
            # plt.savefig('./train_cae.png')
            # plt.draw()
            # plt.pause(0.05)
            # plt.show()



    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(loss_list)+1), np.log10(loss_list))
    plt.ylabel('training error')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(val_list)+1), np.log10(val_list))
    plt.xlabel('epochs')
    plt.ylabel('validation error')

    plt.show()

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
    summary(model, input_size=(DIM, domain_size, domain_size))

    # Save the initial value of the latent variable
    initial = model.encoder(snapshots[:1, :, :, :].to(device, dtype=torch.float)).detach().to(torch.device('cpu'), dtype=torch.double).numpy()
    print("initial latent variable shape : ", initial)
    np.save("latent_initial.npy", initial)

    # Save decoder
    model.decoder.to(device)
    sm = torch.jit.script(model.decoder)
    sm.save('decoder_gpu.pt')

    device = 'cpu'
    model.decoder.to(device)
    sm = torch.jit.script(model.decoder)
    sm.save('decoder.pt')



if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num_epochs', default=30, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning_rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch', '--batch_size', default=50, type=int, help='learning rate')
    parser.add_argument('-dim', '--latent_dim', default=4, type=int, help='learning rate')
    parser.add_argument('-device', '--device', default='True', type=str, help='whether to use cuda')
    parser.add_argument('-i', '--iter', default=2, type=int, help='epoch when visualization runs')
    args = parser.parse_args()

    main(args)
