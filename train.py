import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as F
import argparse
from torch.optim.lr_scheduler import ExponentialLR


# --------------------------------- Arguments --------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('-b', dest='b', type=int)
parser.add_argument('-e', dest='e', type=int)
parser.add_argument('-s', dest='s', type=str)
parser.add_argument('-z', dest='z', type=int)
parser.add_argument('-p', dest='p', type=str)
parser.add_argument('-l', dest='l', type=str)
options = parser.parse_args()
device = torch.device("cpu")
batch_size = options.b
epochs = options.e

# --------------------------------- Loading in the MNIST Dataset --------------------------------- #

# Load the MNIST dataset
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = train_transform

train_set = MNIST(root = "./data/mnist",
                         train = True,
                         download = True,
                         transform = train_transform)
test_set = MNIST(root = "./data/mnist",
                         train = True,
                         download = True,
                         transform = test_transform)

# DataLoader is used to load the dataset for training
train_loader = torch.utils.data.DataLoader(train_set,
                                     batch_size = 2048,
                                     shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set,
                                     batch_size = 2048,
                                     shuffle = False)


# --------------------------------- Creating the MLP Autoencoder Class --------------------------------- #

class MLP4LayerAutoencoder(nn.Module):
    def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
        super(MLP4LayerAutoencoder, self).__init__()
         
        N2 = 392
        self.fc1 = nn.Linear(N_input, N2)  # input = 1x784, output = 1x392
        self.fc2 = nn.Linear(N2, N_bottleneck)  # output = 1xN_bottleneck
        self.fc3 = nn.Linear(N_bottleneck, N2)  # output = 1x392
        self.fc4 = nn.Linear(N2, N_output)  # output = 1x784

        self.type = 'MLP4'
        self.input_shape = (1, 28 * 28)
 
    def forward(self, x):
        return self.decode(self.encode(x))
    
     # Step 6 Bottleneck Interpolation
    def encode(self, x):
        # Encoder
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return x

    def decode(self, x):
        
        # Decoder
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.sigmoid(x)
        return x


# --------------------------------- Model Parameters --------------------------------- #
# Model Initialization
model = MLP4LayerAutoencoder()
 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.3
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-3,
                             weight_decay = 1e-5)
# Scheduler
scheduler = ExponentialLR(optimizer, gamma=0.9)

# --------------------------------- Model Training --------------------------------- #
def train(epochs, optimizer, model, loss_function, train_loader, scheduler,device):
    print("Training...")
    model.train() #Keep track of gradient for backtracking
    losses_train = []
    

    for epoch in range(1,epochs+1):
        print('epoch ', epoch)
        loss_train = 0.0

        for imgs, label in train_loader:
            imgs = imgs.reshape(-1, 28*28)

            imgs = imgs.to(device)
            outputs = model(imgs)
            
            loss = loss_function(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        
        scheduler.step(loss_train)

        losses_train += [loss_train/len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(),epoch, loss_train/len(train_loader)))
    plt.plot(losses_train, label="train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc=1)
    plt.savefig(options.p) # loss.MLP.8.png

train(epochs=epochs, optimizer=optimizer, model=model, loss_function=loss_function, train_loader=train_loader,scheduler=scheduler, device=device )