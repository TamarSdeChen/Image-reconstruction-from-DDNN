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
    return outputs, labels , l1, l2, l3, l4, l5, l6, l7

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
#test_samples = test_samples[torch.randperm(test_samples.size()[0])]
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, sampler=test_samples)

output_test, labels , l1, l2, l3, l4, l5, l6, l7 = calculate_accuracy(model, test_loader, device)
output_test = output_test.to('cpu')
list_layers = [l1, l2, l3, l4, l5, l6, l7]
print('hi iggy, hi tamar')

# # plot non target class results:
# fig = plt.figure()
# for i in range(test_size):
#     plt.subplot(int(test_size/10),10, i + 1)
#     plt.title('label: {}'.format(labels[i]))
#     plt.axis('off')
#     plt.imshow(torch.sqrt(output_test[i][0]), cmap='gray')


# -----------------------------------------------------------------------  best match  -----------------------------:
# plot chosen classes:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fourier_and_normalize(image):
    out_im = torch.fft.fft2(torch.fft.fftshift((image.cpu().detach()))).abs() / torch.max(torch.fft.fft2(torch.fft.fftshift((image.cpu().detach()))).abs())
    return out_im

transform2 = torchvision.transforms.Compose([
    torchvision.transforms.Resize([90, 90]),
    torchvision.transforms.Pad([15, 15]),])
knn_size = 100
zero_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 0][0:knn_size].unsqueeze(0)).float()
one_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 1][0:knn_size].unsqueeze(0)).float()
three_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 3][0:knn_size].unsqueeze(0)).float()
four_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 4][0:knn_size].unsqueeze(0)).float()
five_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 5][0:knn_size].unsqueeze(0)).float()
six_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 6][0:knn_size].unsqueeze(0)).float()
seven_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 7][0:knn_size].unsqueeze(0)).float()
eight_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 8][0:knn_size].unsqueeze(0)).float()
nine_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 9][0:knn_size].unsqueeze(0)).float()
classes_img = ([zero_class,one_class,three_class,four_class,five_class,six_class,seven_class,eight_class,nine_class])
classes_lbl = [0,1,3,4,5,6,7,8,9]
# plot non target class results:
fig = plt.figure()
classes_out = []
for i,im in enumerate(classes_img):
    model.eval()  # put in evaluation mode
    # im_tmp = torch.zeros_like(im[0])
    # im_tmp[im[0] > 0] = 1
    # im_tmp[im_tmp > ]
    output_tmp, _, _, _, _, _, _, _ = model(im[0].to(device))
    plt.subplot(1,9, i + 1)
    plt.title('label: {}'.format(classes_lbl[i]))
    plt.axis('off')
    plt.imshow((output_tmp[0].cpu().detach().numpy()), cmap='gray')
    # plt.imshow(im_tmp,cmap='gray')
    for j in range(knn_size):
        output_tmp[j] = fourier_and_normalize(output_tmp[j])
    classes_out.append(output_tmp.cpu().detach())
# plt.show()

def pearson_coef(output,target):
    eps = 10 ** -7
    x = output
    y = target

    vx = x - torch.mean(x)
    #vx = vx / vx.max() # normalization does not change pearson
    vy = y - torch.mean(y)
    #vy = vy / vy.max() # normalization does not change pearson
    res = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + eps)
    return res


def get_top_indices(list_of_lists, n):
    flattened_list = [element for sublist in list_of_lists for element in sublist]
    sorted_indices = sorted(range(len(flattened_list)), key=lambda i: flattened_list[i], reverse=True)
    top_indices = [divmod(idx, len(list_of_lists[0])) for idx in sorted_indices[:n]]
    return top_indices


import torch.nn.functional as F
# loss = F.mse_loss(, )
from collections import Counter

def find_most_common_row(top_indices):
    rows = [index[0] for index in top_indices]
    counter = Counter(rows)
    most_common_row = counter.most_common(1)[0][0]
    return most_common_row

keys = [0,1,3,4,5,6,7,8,9]
def find_best_match(classes_out, classes_lbl, image, keys):
    pcc_max = -2
    label_max = -2
    pcc_mat = []
    nmse_mat = []
    for i in range(9):
        pcc_vec = []
        nmse_vec = []
        for j in range(knn_size):
            tmp_pcc = pearson_coef(classes_out[i][j], image)
            tmp_nmse = F.mse_loss(classes_out[i][j]/torch.max(classes_out[i][j]), image/torch.max(image))
            pcc_vec.append(tmp_pcc)
            nmse_vec.append(tmp_nmse)
            # if pearson_coef(classes_out[i], image) > pcc_max:
            #     pcc_max = pearson_coef(classes_out[i], image)
            #     label_max = classes_lbl[i]
        pcc_mat.append(pcc_vec)
        nmse_mat.append(nmse_vec)
    top_indices = get_top_indices(pcc_mat, 5)
    label_max = keys[find_most_common_row(top_indices)]
    return label_max,nmse_mat, pcc_mat

# model.eval() # put in evaluation mode
#
# chosen_im = transform2(test_loader.dataset.data[test_loader.dataset.targets == 3][17].unsqueeze(0)).float()[0]
# fig = plt.figure()
# plt.imshow(chosen_im, cmap='gray')
# output_tmp, _, _, _, _, _, _, _ = model(chosen_im.to(device))
# label_max, nmse_mat, pcc_mat = find_best_match(classes_out, classes_lbl, fourier_and_normalize(output_tmp.cpu().detach()),keys)
count_three = 0
label = 6
for i in range(100):
    model.eval()  # put in evaluation mode
    chosen_im = transform2(test_loader.dataset.data[test_loader.dataset.targets == label][knn_size+i].unsqueeze(0)).float()[0]
    output_tmp, _, _, _, _, _, _, _ = model(chosen_im.to(device))
    label_max, nmse_mat, pcc_mat = find_best_match(classes_out, classes_lbl,
                                                   fourier_and_normalize(output_tmp.cpu().detach()), keys)
    if label_max == label:
        count_three += 1
print("the number of correct predictions is: " + str(count_three))
# print("the pcc vector ",pcc_mat)
# print("the nmse vector ",nmse_mat)

"""
# plot chosen classes:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fourier_and_normalize(image):
    out_im = torch.fft.fft2(torch.fft.fftshift((image.cpu().detach()))).abs() / torch.max(torch.fft.fft2(torch.fft.fftshift((image.cpu().detach()))).abs())
    return out_im

transform2 = torchvision.transforms.Compose([
    torchvision.transforms.Resize([90, 90]),
    torchvision.transforms.Pad([15, 15]),])
zero_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 0][0].unsqueeze(0)).float()
one_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 1][0].unsqueeze(0)).float()
three_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 3][0].unsqueeze(0)).float()
four_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 4][0].unsqueeze(0)).float()
five_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 5][0].unsqueeze(0)).float()
six_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 6][0].unsqueeze(0)).float()
seven_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 7][0].unsqueeze(0)).float()
eight_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 8][0].unsqueeze(0)).float()
nine_class = transform2(test_loader.dataset.data[test_loader.dataset.targets == 9][0].unsqueeze(0)).float()
classes_img = ([zero_class,one_class,three_class,four_class,five_class,six_class,seven_class,eight_class,nine_class])
classes_lbl = [0,1,3,4,5,6,7,8,9]
# plot non target class results:
fig = plt.figure()
classes_out = []
for i,im in enumerate(classes_img):
    model.eval()  # put in evaluation mode
    im_tmp = torch.zeros_like(im[0])
    im_tmp[im[0] > 0] = 1
    # im_tmp[im_tmp > ]
    output_tmp, _, _, _, _, _, _, _ = model(im[0].to(device))
    plt.subplot(1,9, i + 1)
    plt.title('label: {}'.format(classes_lbl[i]))
    plt.axis('off')
    output_tmp_norm = output_tmp / torch.max(output_tmp)
    plt.imshow(torch.fft.fft2(torch.fft.fftshift((output_tmp_norm.cpu().detach()))).abs(), cmap='gray')
    # plt.imshow(im_tmp,cmap='gray')
    classes_out.append(fourier_and_normalize(output_tmp.cpu().detach()))
# plt.show()


def pearson_coef(output,target):
    eps = 10 ** -7
    x = output
    y = target

    vx = x - torch.mean(x)
    #vx = vx / vx.max() # normalization does not change pearson
    vy = y - torch.mean(y)
    #vy = vy / vy.max() # normalization does not change pearson
    res = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + eps)
    return res

import torch.nn.functional as F
# loss = F.mse_loss(, )

def find_best_match(classes_out, classes_lbl, image, metric="nmse"):
    pcc_max = -2
    label_max = -2
    pcc_vec = []
    nmse_vec = []
    for i in range(9):
        tmp_pcc = pearson_coef(classes_out[i], image)
        tmp_nmse = F.mse_loss(classes_out[i]/torch.max(classes_out[i]), image/torch.max(image))
        pcc_vec.append(tmp_pcc)
        nmse_vec.append(tmp_nmse)
        if pearson_coef(classes_out[i], image) > pcc_max:
            pcc_max = pearson_coef(classes_out[i], image)
            label_max = classes_lbl[i]

    return label_max,nmse_vec, pcc_vec

model.eval() # put in evaluation mode
chosen_im = transform2(test_loader.dataset.data[test_loader.dataset.targets == 3][10].unsqueeze(0)).float()[0]
fig = plt.figure()
plt.imshow(chosen_im, cmap='gray')
output_tmp, _, _, _, _, _, _, _ = model(chosen_im.to(device))
label_max, nmse_vec, pcc_vec = find_best_match(classes_out, classes_lbl, fourier_and_normalize(output_tmp.cpu().detach()))
print("the prediction for the class is " + str(label_max))
print("the pcc vector ", pcc_vec)
print("the nmse vector ", nmse_vec)
plt.show()

"""