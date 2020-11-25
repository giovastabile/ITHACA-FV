import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from convae import *

NUM_EPOCHS = 40
HIDDEN_DIM = 4
BATCH_SIZE = 20
LEARNING_RATE = 0.001

snapshots = np.load("../../npSnapshots.npy")
n_train = snapshots.shape[1]

# reshape as (train_samples, channel, y, x)
snapshots = snapshots.T
snapshots = snapshots.reshape((n_train, 3, 150, 150))

snapshots = torch.from_numpy(snapshots)
print("snapshots shape: ", snapshots.size())

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device is: ", device)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=snapshots,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

model = AE(HIDDEN_DIM).to(device)

# Loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_list = []
counter = 0

# Train the model
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, snapshot in enumerate(train_loader):

        snapshot = snapshot.to(device, dtype=torch.float)
        # Forward pass
        outputs = model(snapshot)
        loss = criterion(outputs, snapshot)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 4 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

    loss_list.append(loss)

plt.plot(range(NUM_EPOCHS), loss_list)
plt.show()
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')