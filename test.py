import numpy

import optic_network
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import time
import os
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter

# function to calculate accuracy of the model
def calculate_accuracy(model, dataloader, device=torch.device('cuda:0')):
    model.eval() # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10,10], int)
    outputs = []
    labels= []
    imagess = []
    with torch.no_grad():
        for data in dataloader:
            images, label = data
            images = images.to(device)
            label = label.to(device)
            output, l1, l2, l3, l4, l5, l6, l7 = model(images)
            outputs.append(output)
            labels.append(label)
            imagess.append(images)
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        imagess = torch.cat(imagess)


    return outputs, labels , imagess

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
test_size = 180
batch_size_test = 2*test_size

test_set = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)
idx_target = test_set.targets == 2
idx_else = (test_set.targets != 2)  # | (train_set.targets == 3)
test_samples = torch.cat((idx_target.nonzero()[0:test_size], idx_else.nonzero()[0:test_size])).squeeze()
# test_samples = [  5,  16,  25,  28,  76,  82, 109, 117, 120, 122, 143, 159, 161, 171,
#         178, 180, 187, 189, 190, 199, 213, 220, 233, 252, 253, 262, 268, 277,
#         308, 317, 318, 325, 339, 347, 360, 365, 375, 378, 381, 385,   0,   1,
#           2,   3,   4,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  17,
#          18,  19,  20,  21,  22,  23,  24,  26,  27,  29,  30,  31,  32,  33,
#          34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
#          48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
#          62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
#          77,  78,  79,  80,  81,  83,  84,  85]
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, sampler=test_samples)

# test_loader = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('../data', train=False, download=True,
#                                    transform=transform), batch_size=batch_size_test, shuffle=True)

# test_accuracy, confusion_matrix = optic_network.calculate_accuracy(model, test_loader, device)
output_test, labels , images = calculate_accuracy(model, test_loader, device)
output_test = output_test.to('cpu')

print('hi iggy, hi tamar')

# # plot target class results:
# fig = plt.figure(layout="constrained")
# for i in range(9):
#     plt.subplot(3,3, i + 1)
#     plt.title('label: {}'.format(labels[i]))
#     plt.axis('off')
#     plt.imshow(torch.sqrt(output_test[i+50][0]), cmap='gray')
# #
# # plot non target class results:
# fig = plt.figure(layout="constrained")
# for i in range(9):
#     plt.subplot(3,3, i + 1)
#     plt.title('label: {}'.format(labels[test_size+i+30]))
#     plt.axis('off')
#     tmp_out = torch.sqrt(output_test[test_size + i+30][0])
#     # tmp_out = (tmp_out-torch.min(tmp_out))/(torch.max(tmp_out) - torch.min(tmp_out))
#     tmp_out[tmp_out > 0.15] = 255
#     # print(torch.min(tmp_out))
#
#     #plt.imshow(torch.sqrt(output_test[test_size + i][0]), cmap='gray')
#     plt.imshow(torch.sqrt(tmp_out), cmap='gray')
z = 25
transformg = T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2))
fig = plt.figure(layout="constrained")
for i in range(2):
    plt.subplot(4,1, i + 1+2)
    # plt.title('label: {}'.format(labels[test_size+i]))
    plt.axis('off')
    tmp_out = torch.sqrt(output_test[test_size + i+z][0])
    # tmp_out = (tmp_out-torch.min(tmp_out))/(torch.max(tmp_out) - torch.min(tmp_out))
    tmp_out[tmp_out > 0.17] = 255
    # print(torch.min(tmp_out))

    #plt.imshow(torch.sqrt(output_test[test_size + i][0]), cmap='gray')
    plt.imshow(gaussian_filter(tmp_out,sigma=1), cmap='gray')
for i in range(2):
    plt.subplot(4, 1, i + 1)
    # plt.title('label: {}'.format(labels[i]))
    plt.axis('off')
    tmp_out = torch.sqrt(output_test[i+z][0])
    # tmp_out = (tmp_out-torch.min(tmp_out))/(torch.max(tmp_out) - torch.min(tmp_out))
    tmp_out[tmp_out > 0.13] = 255
    plt.imshow(gaussian_filter(tmp_out,sigma=1), cmap='gray')

fig = plt.figure(layout="constrained")
for i in range(2):
    plt.subplot(4,1, i + 1+2)
    # plt.title('label: {}'.format(labels[test_size+i]))
    plt.axis('off')
    tmp_out = torch.sqrt(images[test_size + i+z][0].cpu())
    # tmp_out = (tmp_out-torch.min(tmp_out))/(torch.max(tmp_out) - torch.min(tmp_out))
    # tmp_out[tmp_out > 0.15] = 255
    # print(torch.min(tmp_out))

    #plt.imshow(torch.sqrt(output_test[test_size + i][0]), cmap='gray')
    plt.imshow(torch.sqrt(tmp_out), cmap='gray')
for i in range(2):
    plt.subplot(4, 1, i + 1)
    # plt.title('label: {}'.format(labels[i]))
    plt.axis('off')
    plt.imshow(torch.sqrt(images[i+z][0].cpu()), cmap='gray')

#
plt.show()


# fig = plt.figure()
# for i in range(20):
#     plt.subplot(5,10, i + 1)
#     plt.title('label: {}'.format(labels[i]))
#     plt.imshow(torch.sqrt(output_test[i][0]), cmap='gray')
#
# plt.show()
# i = 1
# for image in list_layers:
#     plt.subplot(1,len(list_layers)+1, i)
#     plt.title('layer: {}'.format(i))
#     plt.imshow((image[16][0].abs() ** 2).cpu(), cmap='gray')
#     i +=1
# plt.subplot(1,len(list_layers)+1, i)
# plt.title('layer: {}'.format(i))
# plt.imshow(torch.sqrt((output_test[16][0]).cpu()), cmap='gray')
#
# plt.show()



phase = model.phase3.weights.cpu().detach().numpy()
fig = plt.figure()
plt.imshow(numpy.mod(phase,torch.pi*2))
#plt.title('third Phase Mask Layer')
#plt.colorbar()
plt.show()