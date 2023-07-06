# imports
import os
import time
import torch
import random
import torchvision
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import rotate, AffineTransform, warp
from optic_network import RS_estimation, customLoss, OpticModel
random_index = 0

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for tensorboard
    # writer = SummaryWriter(log_dir=f"log_dir")

    # define optic simulation parameters - tbd from article
    pixel_size = 0.4 * (10 ** -3)
    layer_size = 120
    L1 = pixel_size * layer_size # side length
    lambda_in = 0.75 * (10 ** -3)  # wavelength in m
    z = 20 * (10 ** -3)  # propagation dist (m)

    # RS approximation
    # H = RS_estimation(layer_size, L1, lambda_in, z)
    H = RS_estimation(layer_size, L1, lambda_in, z)

    # hyper parameters to be decided from paper:
    target_label = 2
    epochs = 1000  # need to change to 1000 after !!
    batch_size_train = 60
    learning_rate = 0.03

    # loss criterion
    criterion = customLoss()

    # build our model and send it to the device
    model = OpticModel(H, layer_size).to(device)  # no need for parameters as we already defined them in the class

    # optimizer - Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # each diffractive layer contains 120x120 diffractive pixels - ask Matan
    # each pixel/neuron size is 0.4mm
    # data augmentations (for each image)
    # linearly upscale 28x28 to 90x90
    # random transformations:
    # rotation by an angle within [-10,10]
    # random scaling by a factor within [0.9,1.1]
    # random shift in each lateral direction by an amount of [-2.13lambda,2.13lambda] -
    # ask Matan (horizontal shift is randomly
    # sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled
    # in the range -img_height * b < dy < img_height * b. Will not translate by default.)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([90, 90]),
        torchvision.transforms.RandomRotation([-50, 50]),
        torchvision.transforms.RandomAffine(degrees=[0.9, 1.1], translate=(2.13 * lambda_in, 2.13 * lambda_in)),
        torchvision.transforms.Pad([15, 15]),
        torchvision.transforms.ToTensor(),
    ])
    # transform for G_K
    transform_1 = torchvision.transforms.Compose([
        torchvision.transforms.Resize([90, 90]),
        torchvision.transforms.Pad([15, 15]),

    ])

    # 48000 train , 12000 validation, 10000 test - ask Matan
    # data loader sampler to not sample all MNIST
    train_set = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_samples = torch.randperm(train_set.data.shape[0])[:50]

    #make idx vector of our target and not target
    idx_target = train_set.targets == 2
    idx_else = (train_set.targets != 2) #| (train_set.targets == 3)
    train_samples = torch.cat((idx_target.nonzero()[0:5958],idx_else.nonzero()[0:11916])).squeeze()
    #print(" data chosen:" + str([train_samples]))
    train_samples = train_samples[torch.randperm(train_samples.size()[0])]



    # # 50% from label==target sampler 5.55% from each not target
    # class_weights = [1 / 18, 1 / 18, 1 / 2, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18]
    # sample_weights = [0] * len(train_set)
    #
    # for idx, (data, label) in enumerate(train_set):
    #     class_weight = class_weights[label]
    #     sample_weights[idx] = class_weight
    #
    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, sampler=train_samples)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)

    ## if not using sampler uncomment this row:
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train)

    # list of target class from data loader to choose random target digit G_k
    train_labels = train_loader.dataset.targets
    # print(train_labels[np.array(train_samples)])
    target_dataset = train_loader.dataset.data[train_labels == target_label]
    n_target_samples = len(target_dataset)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # randomly choose G_k from target_label in training set: (for each batch. using this in loss calculations)
            # right now the random sample get from the big dataset and not from the batch itself !!!
            # Get a random sample
            random_index = int(np.random.random() * n_target_samples)
            G_k = (transform_1(target_dataset[random_index].unsqueeze(0).type(torch.float)/255).to(device))
            # tensorboard
            # GK = torchvision.utils.make_grid(torch.abs(G_k) ** 2)
            # writer.add_image('G_K', GK)

            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)


            # forward + backward + optimize
            outputs, l1,l2,l3,l4,l5,l6,l7 = model(inputs)  # forward pass
            # l1_img = torchvision.utils.make_grid(torch.abs(l1) ** 2)
            # l2_img = torchvision.utils.make_grid(torch.abs(l2) ** 2)
            # l3_img = torchvision.utils.make_grid(torch.abs(l3) ** 2)
            # l4_img = torchvision.utils.make_grid(torch.abs(l4) ** 2)
            # l5_img = torchvision.utils.make_grid(torch.abs(l5) ** 2)
            # l6_img = torchvision.utils.make_grid(torch.abs(l6) ** 2)
            # l7_img = torchvision.utils.make_grid(torch.abs(l7) ** 2)
            # num_epoch = str(epoch)
            # writer.add_image(num_epoch + 'after propFirst', l1_img)
            # writer.add_image(num_epoch + 'after phase1', l2_img)
            # writer.add_image(num_epoch + 'after prop2', l3_img)
            # writer.add_image(num_epoch + 'after phase2', l4_img)
            # writer.add_image(num_epoch + 'after prop3', l5_img)
            # writer.add_image(num_epoch + 'after phase3', l6_img)
            # writer.add_image(num_epoch + 'after propLast', l7_img)

            # calculate the loss: +loss for target label, anf -loss for all other. choose G_k randomly from inputs
            loss = criterion(inputs, outputs, labels, target_label,G_k)
            # backward pass
            optimizer.zero_grad()   # clean the parameter gradients
            loss.backward()         # backpropagation
            optimizer.step()        # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        #running_loss /= len(train_loader)
        # writer.add_scalar('Train Loss', running_loss, epoch)

        # Calculate training/test set accuracy of the existing model
        # train_accuracy, _ = calculate_accuracy(model, train_loader, device)
        # test_accuracy, _ = calculate_accuracy(model, test_loader, device)

        log = "Epoch: {} | Loss: {:.4f} | ".format(epoch, running_loss)
        # with f-strings
        # log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training accuracy: {train_accuracy:.3f}% | Test accuracy: {test_accuracy:.3f}% |"
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        # with f-strings
        # log += f"Epoch Time: {epoch_time:.2f} secs"
        print(log)

        # save model
        if epoch % 20 == 0:
            print('==> Saving model ...')
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(state, './checkpoints/all_samples_1to9Ratio_withNoAbs111.pth')

    print('==> Finished Training ...')


if __name__ == "__main__":
    train()
