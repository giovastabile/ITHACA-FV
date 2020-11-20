import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

snapshots = np.load("../npSnapshots.npy")
n_train = snapshots.shape[1]

# reshape as (train_samples, channel, y, x)
snapshots = snapshots.T
snapshots = snapshots.reshape((n_train, 3, 150, 150))

def plot_snapshot(snapshot, idx_train, idx_coord):
    x, y = np.meshgrid(np.arange(150), np.arange(150))
    z = snapshot[idx_train, idx_coord, x, y]
    plt.contourf(x, y, z)
    plt.show()

plot_snapshot(snapshots, 80, 0)

snapshots = torch.from_numpy(snapshots)
print("snapshots shape: ", snapshots.size())

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 100
hidden_dim = 4
batch_size = 100
learning_rate = 0.001

# Convolutional neural network (two convolutional layers)
class Encoder(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        print(out.size())
        out = self.fc(out)
        return out

class Decoder(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        print(out.size())
        out = self.fc(out)
        return out

model = ConvNet(hidden_dim).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')