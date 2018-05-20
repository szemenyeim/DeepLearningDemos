import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, \
    DataLoader
import visualize

# Plotter
plotter = visualize.LinePlotter("CVSDemo")

# dataset and network sizes
N,D_in,H,D_out = 128,1000,100,10

# Different LR values
lrs = [4e-3, 3e-3, 1e-3, 5e-4, 1e-4]

# Simple 2 layer MLP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# create in and output
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Mean Squared Error Loss
criterion = torch.nn.MSELoss(size_average=False)

# Dataloader
loader = DataLoader(TensorDataset(x, y),
                    batch_size=16)

for i in range(5):

    # Make sure that all runs are comparable
    torch.manual_seed(1)

    # create net
    model = Net()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lrs[i])

    # Run 200 epochs
    for epoch in range(200):
        err = 0
        # For every minibatch:
        for x_b, y_b in loader:

            # Run network and compute loss
            y_pred = model(x_b)
            loss = criterion(y_pred,y_b)

            # Compute gradients
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()

            # compute error
            err += loss.item()
        #print err
        plotter.plot("LR=%.2E" % lrs[i], "Error", epoch,err)

