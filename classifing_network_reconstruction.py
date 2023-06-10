import optic_network
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model, calculate accuracy and confusion matrix

# define optic simulation parameters - tbd from article
# L1 = 0.5  # side length
# lambda_in = 0.75 * (10 ** -3)  # wavelength in m
# k = 2 * np.pi / lambda_in  # wavenumber
# z = 20 * (10 ** -3)  # propagation dist (m)
# layer_size = 120

pixel_size = 0.4 * (10 ** -3)
layer_size = 120
L1 = pixel_size * layer_size # side length
lambda_in = 0.75 * (10 ** -3)  # wavelength in m
z = 20 * (10 ** -3)  # propagation dist (m)
k = 2 * np.pi / lambda_in  # wavenumber

# RS approximation -
H = optic_network.RS_estimation(layer_size, L1, lambda_in, z)  ##RS approximation


model = optic_network.OpticModel(H, layer_size).to(device)
# state = torch.load('./checkpoints/try_with_5000t_4000nt_samples_1000_epoch.pth', map_location=device) #goodtrain
state = torch.load('./checkpoints/all_samples_1to9Ratio.pth', map_location=device)
model.load_state_dict(state['net'])
# note: `map_location` is necessary if you trained on the GPU and want to run inference on the CPU

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([90, 90]),
    torchvision.transforms.Pad([15, 15]),
    torchvision.transforms.ToTensor(),
])
test_size = 60
batch_size_test = 2*test_size

test_set = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)
idx_target = test_set.targets == 2
idx_else = (test_set.targets != 2)  # | (train_set.targets == 3)
test_samples = (idx_else.nonzero()).squeeze()
test_samples = test_samples[torch.randperm(test_samples.size()[0])]
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, sampler=test_samples)

output_test, labels , l1, l2, l3, l4, l5, l6, l7 = calculate_accuracy(model, test_loader, device)
output_test = output_test.to('cpu')
list_layers = [l1, l2, l3, l4, l5, l6, l7]
print('hi iggy, hi tamar')

# plot non target class results:
fig = plt.figure()
for i in range(test_size):
    plt.subplot(int(test_size/10),10, i + 1)
    plt.title('label: {}'.format(labels[i]))
    plt.axis('off')
    plt.imshow(torch.sqrt(output_test[i][0]), cmap='gray')

plt.show()
# ----------------------------------------------------------------------- VIT network -----------------------------:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
torch.backends.cudnn.enabled = False

from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.y[idx]
        return features, target
samples = torch.randperm(len(output_test))
test_loader_train = torch.utils.data.DataLoader(CreateDataset(output_test,labels), batch_size=batch_size_test, sampler=samples[0:8000])
test_loader_test = torch.utils.data.DataLoader(CreateDataset(output_test,labels), batch_size=batch_size_test, sampler=samples[8000:-1])
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(27*27*20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 27*27*20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(test_loader_train.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(test_loader_train):
    optimizer.zero_grad()
    # send them to device
    data = data.to(device)
    target = target.type(torch.LongTensor).to(device)
    output = network(data)
    # output = output.to(device)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(test_loader_train.dataset),
        100. * batch_idx / len(test_loader_train), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(test_loader_train.dataset)))
      if not os.path.isdir('results'):
          os.mkdir('results')
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader_test:
      data = data.to(device)
      output = network(data)
      target = target.type(torch.LongTensor).to(device)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader_test.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader_test.dataset),
    100. * correct / len(test_loader_test.dataset)))

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
# test_loader_train = torch.utils.data.DataLoader(output_test[:][0], batch_size=batch_size_test, sampler=test_samples[0:8000])
# test_loader_test = torch.utils.data.DataLoader(output_test[:][0], batch_size=batch_size_test, sampler=test_samples[8000:-1])
#
# image_size = 120
# channel_size = 1
# patch_size = 7
# embed_size = 512
# num_heads = 8
# classes = 10
# num_layers = 3
# hidden_size = 256
# dropout = 0.2
#
# BATCH_SIZE = 64
# LR = 5e-5
# NUM_EPOCHES = 25
#
# model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)
# for img, label in trainloader:
#     img = img.to(device)
#     label = label.to(device)
#
#     print("Input Image Dimensions: {}".format(img.size()))
#     print("Label Dimensions: {}".format(label.size()))
#     print("-" * 100)
#
#     out = model(img)
#
#     print("Output Dimensions: {}".format(out.size()))
#     break
#
# criterion = nn.NLLLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
#
# loss_hist = {}
# loss_hist["train accuracy"] = []
# loss_hist["train loss"] = []
#
# for epoch in range(1, NUM_EPOCHES + 1):
#     model.train()
#
#     epoch_train_loss = 0
#
#     y_true_train = []
#     y_pred_train = []
#
#     for batch_idx, (img, labels) in enumerate(trainloader):
#         img = img.to(device)
#         labels = labels.to(device)
#
#         preds = model(img)
#
#         loss = criterion(preds, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         y_pred_train.extend(preds.detach().argmax(dim=-1).tolist())
#         y_true_train.extend(labels.detach().tolist())
#
#         epoch_train_loss += loss.item()
#
#     loss_hist["train loss"].append(epoch_train_loss)
#
#     total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x == y])
#     total = len(y_pred_train)
#     accuracy = total_correct * 100 / total
#
#     loss_hist["train accuracy"].append(accuracy)
#
#     print("-------------------------------------------------")
#     print("Epoch: {} Train mean loss: {:.8f}".format(epoch, epoch_train_loss))
#     print("       Train Accuracy%: ", accuracy, "==", total_correct, "/", total)
#     print("-------------------------------------------------")
#
# plt.plot(loss_hist["train accuracy"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show()