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

# --------------------------------- Step 4: Test Your Autoencoder --------------------------------- #
# Load the calculated weights
model.load_state_dict(torch.load(options.l))

# Set model to eval mode
model.eval()

with torch.no_grad(): #Disable gradient calculations for testing
      for i, item in enumerate(test_loader):
            imgs, labels = item

            imgs = imgs.to(device=device)
            imgs = imgs.type(torch.float32)

            # The original input image
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            imgs = imgs.reshape([-1, 28, 28])
            plt.imshow(imgs[0], cmap='gray')

            
            # Output from the model (denoised)
            imgs = imgs.reshape(-1,28*28)
            result = model(imgs)
            result = result.reshape([-1, 28, 28])
            f.add_subplot(1,2,2)
            plt.imshow(result[0], cmap='gray')
            plt.show()

            # Only show 2 images so the rest of the code can load
            if(i == 1):
                break

# --------------------------------- Step 5: Image Denoising --------------------------------- #
with torch.no_grad(): #Disable gradient calculations for testing
      for i, item in enumerate(test_loader):
            imgs, labels = item

            imgs = imgs.to(device=device)
            imgs = imgs.type(torch.float32)

            # The original input image
            f = plt.figure()
            f.add_subplot(1, 3, 1)
            imgs = imgs.reshape([-1, 28, 28])
            plt.imshow(imgs[0], cmap='gray')

            # Adding noise to the images 
            noise = torch.rand(imgs.shape).to(device)
            noisy_img = imgs + noise
            f.add_subplot(1,3,2)
            plt.imshow(noisy_img[0], cmap='gray')
            
            # Output from the model (denoised)
            noisy_img = noisy_img.reshape(-1,28*28)
            result = model(noisy_img)
            result = result.reshape([-1, 28, 28])
            f.add_subplot(1, 3, 3)
            plt.imshow(result[0], cmap='gray')
            plt.show()

            # Only show 2 images so the rest of the code can load
            if(i == 1):
                break


# --------------------------------- Step 6: Bottle Neck Interpolation --------------------------------- #

class InterpolationModule(nn.Module):
    def __init__(self, model):
        super(InterpolationModule, self).__init__()
        self.model = model

    def interpolate(self, tensor1, tensor2, n_steps):
        # Linearly interpolate between tensor1 and tensor2
        interpolations = []
        for alpha in torch.linspace(0, 1, n_steps):
            interpolated_tensor = alpha * tensor1 + (1 - alpha) * tensor2
            interpolations.append(interpolated_tensor)
        return interpolations

    def plot_interpolations(self, tensor1, tensor2, n_steps):
        interpolations = self.interpolate(tensor1, tensor2, n_steps) #Receive a list of interpolations

        # Pass each interpolated tensor through the decode method and plot the results
        import matplotlib.pyplot as plt
        for i, interpolated_tensor in enumerate(interpolations):
            decoded_image = self.model.decode(interpolated_tensor) #Decode the tensor
            decoded_image = decoded_image[0].detach().numpy().reshape(28, 28) #Decoded image

            plt.subplot(1, n_steps, i + 1)
            plt.imshow(decoded_image, cmap='gray')
            plt.axis('off')
        plt.show()

interpolation_module = InterpolationModule(model)


first_two_images = []

# Retrieve two images from the train loader
for batch_idx, (images, labels) in enumerate(train_loader):
    if len(images) >= 2:
        first_two_images.append(images[0])  # Append the first image in the batch
        first_two_images.append(images[1])  # Append the second image in the batch
        break  # Break after collecting the first two images

# Reshape the two images before encoding
first_two_images[0]= first_two_images[0].reshape(-1,28*28)
first_two_images[1]= first_two_images[1].reshape(-1,28*28)

bottleneck1 = model.encode(first_two_images[0])
bottleneck2 = model.encode(first_two_images[1])

n_steps = 10
interpolation_module.plot_interpolations(bottleneck1, bottleneck2,n_steps)