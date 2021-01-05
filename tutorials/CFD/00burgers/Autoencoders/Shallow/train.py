import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
import argparse
from shallow import *
import clock

domain_size = 60

def main(args):
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.latent_dim
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    DIM = 2
    ORDER = 2

    snapshots = np.load("/content/drive/MyDrive/Colab/npSnapshots.npy")
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

    model = AE(input_dim=DIM*domain_size**2, output_dim=DIM*domain_size**2, hidden_dim_enc=8000, hidden_dim_dec=33000, latent_dim=HIDDEN_DIM, domain_size=domain_size, scale=(0,max_sn), use_cuda=args.device).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in model.parameters():
    #     reg_loss += l1_crit(param)

    # factor = 0.005
    # criterion += factor * reg_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

    loss_list = []
    val_list = []
    loss = 1
    changed_to_SGD = True
    scheduler = None
    warming = True

    # Train the model
    total_step = len(train_loader)
    for epoch in range(1, NUM_EPOCHS):
        # if epoch > 50:
        #     array_loss = np.log10(np.array(loss_list[-20:]))
        #     print(np.abs(np.var(array_loss)))
        #     if np.abs(np.var(array_loss)) > 0.025:
        #         optimizer = torch.optim.SGD(model.parameters(), lr=0.5*optimizer.param_groups[0]['lr'], momentum=0.9)
        #         print("CHANGE LR: ", optimizer.param_groups[0]['lr'])
        # if warming and epoch > 300:
        #     print("FINISHED WARMING PERIOD")
        #     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        #     warming = False

        for snapshot in train_loader:

            snapshot = snapshot.to(device, dtype=torch.float)
            # Forward pass
            outputs = model(snapshot)
            loss = criterion(outputs.reshape((-1, DIM, domain_size, domain_size)), snapshot*(max_sn-min_sn)+min_sn)
            # loss = torch.norm(outputs.reshape((-1, DIM, domain_size, domain_size))-snapshot, p=ORDER)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # change to SGD
            # if changed_to_SGD and loss < 2.e-5:
            #     print("CHANGE TO SGD")
            #     optimizer = torch.optim.SGD(model.parameters(), lr=0.1*LEARNING_RATE, momentum=0.5)
            #     # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
            #     changed_to_SGD = False

        if scheduler:
            scheduler.step(loss)
        loss_list.append(loss.detach().cpu().numpy())

        # validation
        validation = validation_snap[:].to("cuda", dtype=torch.float)
        outputs_val = model(validation)
        diff = torch.abs(outputs_val.reshape((-1, 2, domain_size, domain_size))-validation)
        # print(diff.reshape(-1, 2*domain_size**2).size())
        index = torch.argmax(diff.reshape(-1, 2*domain_size**2), dim=1)
        # print(index, index // 7200)
        loss_val = torch.max(diff)
        val_list.append(loss_val.detach().cpu().numpy())

        # plt.ion()
        if epoch % args.iter == 0:

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}\n Validation, Loss: {:.6f}'.format(epoch, NUM_EPOCHS, epoch, total_step, loss.item(), loss_val))

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
    plt.plot(range(1, NUM_EPOCHS), np.log10(loss_list))
    plt.ylabel('training error')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, NUM_EPOCHS), np.log10(val_list))
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
