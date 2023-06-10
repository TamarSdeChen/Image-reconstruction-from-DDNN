import optic_network
import UNET
import time
import torchvision
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn as nn
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import time
import math
from PIL import Image
import glob
from IPython.display import display



torch.manual_seed(0)
np.random.seed(0)

###### test unet
def calculate_accuracy_unet(model, dataloader, device=torch.device('cuda:0')):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10,10], int)
    outputs = []
    labels= []
    with torch.no_grad():
        for data in dataloader:
            images, label = data
            images = images.to(device)
            label = label.to(device)
            output = model(images)
            outputs.append(output)
            labels.append(label.cpu())
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
    return outputs.detach(), labels.detach()

# function to calculate accuracy of the model
def calculate_accuracy(model, dataloader, device=torch.device('cuda:0')):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10,10], int)
    outputs = []
    labels= []
    with torch.no_grad():
        for data in dataloader:
            images, label = data
            images = images.to(device)
            label = label.to(device)
            output, l1, l2, l3, l4, l5, l6, l7 = model(images)
            outputs.append(output)
            labels.append(label)
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
    return outputs.detach(), labels , l1, l2, l3, l4, l5, l6, l7

# DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOAD TRAINED OPTIOPTIC NETWORK
pixel_size = 0.4 * (10 ** -3)
layer_size = 120
L1 = pixel_size * layer_size # side length
lambda_in = 0.75 * (10 ** -3)  # wavelength in m
z = 20 * (10 ** -3)  # propagation dist (m)
k = 2 * np.pi / lambda_in  # wavenumber

# RS approximation -
H = optic_network.RS_estimation(layer_size, L1, lambda_in, z)

# OPTIC MODEL
model = optic_network.OpticModel(H, layer_size).to(device)
# state = torch.load('./checkpoints/try_with_5000t_4000nt_samples_1000_epoch.pth', map_location=device) #goodtrain
state = torch.load('./checkpoints/all_samples_1to9Ratio.pth', map_location=device)
model.load_state_dict(state['net'])

# create the test data det
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([90, 90]),
    torchvision.transforms.Pad([15, 15]),
    torchvision.transforms.ToTensor(),
])
test_set = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)

# create the data loader -  ONLY NOT TARGET EXAMPLES
test_size = 60
batch_size_test = 2*test_size
idx_target = test_set.targets == 2
idx_else = (test_set.targets != 2)
test_samples = (idx_else.nonzero()).squeeze()
# test_samples = test_samples[torch.randperm(test_samples.size()[0])]
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, sampler=test_samples)

# forward pass in the trained model
output_test, labels_optic , l1, l2, l3, l4, l5, l6, l7 = calculate_accuracy(model, test_loader, device)
#output_test = output_test.to('cpu')
list_layers = [l1, l2, l3, l4, l5, l6, l7]
print('hi iggy, hi tamar')

############################### UNET ###########################################################
# create thr dataset to the UNET:
#inputs: UNET input is the OPTIC NETWORK output not target example
#labels: the original MNIST example before the forward in the model

from torch.utils.data import Dataset
class CreateDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target


# transform for the original MNIST images to 112X112 as the output from the unet
transform_1 = transforms.Compose([
        torchvision.transforms.Resize([90, 90]),
        torchvision.transforms.Pad([11, 11]),

    ])

test_set_resize = transform_1(test_set.data[test_samples]).unsqueeze(1).type(torch.float)
samples = torch.randperm(len(output_test))
test_loader_train = torch.utils.data.DataLoader(CreateDataset(output_test,test_set_resize), batch_size=batch_size_test, sampler=samples[0:8000])
test_loader_test = torch.utils.data.DataLoader(CreateDataset(output_test,test_set_resize), batch_size=batch_size_test, sampler=samples[8000:-1])
optic_net_outputs = output_test[samples[8000:-1]]
labels_for_real = labels_optic[samples[8000:-1]]
# make a instance of the UNET
model = UNET.UNET_MODEL().to(device)
#model = UNET(in_channels=3, out_channels=1).to(DEVICE)
#hyper parameter for UNET
lr = 1e-4
batch_size_train = 256
n_epochs = 1000
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# log_interval = 50
# train_losses = []
# train_counter = []
# test_losses = []
#test_counter = [i*len(test_loader_train.dataset) for i in range(n_epochs + 1)]





# load trained unet
model_unet = UNET.UNET_MODEL().to(device)
state = torch.load('./checkpoints_small_unet/epoch_1000', map_location=device)
model_unet.load_state_dict(state['net'])

# test UNET
output_test_unet, labels = calculate_accuracy_unet(model_unet, test_loader_test, device)
output_test_unet = output_test_unet.to('cpu')
# plot target class results:
fig = plt.figure()
x = 35


for i in range(3):
    plt.subplot(3,3, i + 1)
    plt.axis('off')
    # plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.imshow(labels[i+x][0], cmap='gray')
    plt.title('label: {}'.format(labels_for_real[i+x]))


    #plt.subplot(2, 20, i + 1)
    # plt.imshow(labels_for_real[i][0], cmap='gray')
for i in range(3):
    plt.subplot(3,3, i+3 + 1)
    plt.axis('off')
    tmp_out = torch.sqrt(optic_net_outputs[i+x][0].detach().cpu())
    tmp_out[tmp_out > 0.15] = 255
    plt.imshow(tmp_out, cmap='gray')
# plot target class results:
for i in range(3):
    plt.subplot(3,3, i+6 + 1)
    plt.axis('off')
    plt.imshow(output_test_unet[i+x][0], cmap='gray')
#plt.figtext(0.43, 0.88, "Original MNIST images", fontsize=14, weight='bold')
#plt.figtext(0.44, 0.58, "Images after DDNN", fontsize=14, weight='bold')
#plt.figtext(0.40, 0.31, "Reconstructed images from UNET", fontsize=14, weight='bold')

plt.show()