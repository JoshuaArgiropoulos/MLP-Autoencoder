# MLP-Autoencoder
This Python script is designed to train and evaluate an MLP (Multi-Layer Perceptron) Autoencoder on the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits (0-9). The autoencoder aims to compress MNIST images into a lower-dimensional representation (bottleneck) and then reconstruct them as closely as possible to the original images. This process is beneficial for learning efficient data encodings and has applications in noise reduction, data visualization, and feature extraction.

The script includes several key components:

Argument Parsing: Allows for customizable hyperparameters such as batch size, number of epochs, save paths for the model and plots, and more.
MNIST Dataset Loading: Utilizes torchvision's dataset utilities to load and transform the MNIST dataset for training and testing.
MLP4LayerAutoencoder Class: Defines the architecture of the autoencoder with four fully connected layers. It includes an encoder that reduces the input dimension to a bottleneck layer, and a decoder that reconstructs the image from the bottleneck representation.
Training Procedure: Implements the training loop, including forward pass, loss calculation, backpropagation, and optimizer step. It also applies a learning rate scheduler to adjust the learning rate over epochs.
Evaluation and Visualization: After training, the script loads the saved model weights and evaluates the model on the test dataset, showcasing the original images, their noisy versions (for denoising), and the autoencoder's reconstructions.
Bottleneck Interpolation: Demonstrates a method to interpolate between two digit representations in the bottleneck space, visualizing how the autoencoder transitions between these digits.

Usage
To run the script, you need to specify several command-line arguments for configuration, such as batch size (-b), number of epochs (-e), and paths for saving the model (-l) and loss plots (-p). An example command to start training might look like this:

python mnist_autoencoder.py -b 64 -e 20 -p loss_plot.png -l model.pth
