# this is the main file of the project
# imports
#!pip install torchmetrics
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# writer_2 = SummaryWriter(log_dir=f"log_dir")

import time
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
from skimage.transform import rotate, AffineTransform, warp
#from torchmetrics import PearsonCorrCoef
#import kornia
# conda install -c conda-forge torchmetrics
class PhaseMaskLayer(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.size_in = size_in
        weights = torch.ones([120, 120])
        # initialize weights and biases - tbd by article
        # initialize to zero's -  no phase at the start
        torch.nn.init.normal_(weights, mean=0.0, std=1.0)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.



    def forward(self, u1):
        return u1 * torch.exp(-1j*self.weights)

## implementation of Rayleigh-Sommerfeld propogation layer
class PropTF(nn.Module):
  def __init__(self, H):
    # H - transfer function with RS or Frensel approximation
    super().__init__() ## inherit nn.module method
    self.H = H

  def forward(self, u1):
    ## numpy to pytorch -> work with tensor to use GPU
    # u1 - tensor source plane field
    U1 = torch.fft.fft2(torch.fft.fftshift(u1))
    U2 = self.H * U1
    u2 = torch.fft.ifftshift(torch.fft.ifft2(U2))
    return u2  # inverse fft, center obs field


class OpticModel(nn.Module):
    def __init__(self, H, size_x):
        super(OpticModel, self).__init__()
        self.propFirst = PropTF(H)
        self.phase1 = PhaseMaskLayer(size_x)
        self.prop2 = PropTF(H)
        self.phase2 = PhaseMaskLayer(size_x)
        self.prop3 = PropTF(H)
        self.phase3 = PhaseMaskLayer(size_x)
        self.propLast = PropTF(H)

    def forward(self, x):
        x1 = self.propFirst.forward(x)
        x2 = self.phase1.forward(x1)
        x3 = self.prop2.forward(x2)
        x4 = self.phase2.forward(x3)
        x5 = self.prop3.forward(x4)
        x6 = self.phase3.forward(x5)
        x7 = self.propLast.forward(x6)
        x8 = torch.abs(x7) ** 2
        return x8, x1, x2, x3, x4, x5, x6, x7

def RS_estimation(size_in, L, lambda_in, z, device=torch.device('cuda:0')):
  # implementation of Fresnel propagation
  # propagation - transfer function approach
  # assumes same x and y side lengths and
  # uniform sampling
  # u1 - source plane field
  # size_in = M -> [M,~]=size(u1) (pixel num=120)
  # dx = 0.4mm (pixel size / neuron size)
  # dx = L/size_in(M)
  # L - source and observation plane side length
  # lambda - wavelength
  # z - propagation distance
  # u2 - observation plane field
  dx=L/size_in #sample interval
  k=2*np.pi/lambda_in #wavenumber
  #freq coords - FROM -1/(2*dx) TO 1/(2*dx)-1/L IN 1/L steps
  fx=np.arange(  (-1/(2*dx))  ,  (1/(2*dx)-1/L)+(1/L)  ,  (1/L)  ) # check if size is ok (+-1)
  FX,FY=np.meshgrid(fx,fx)
  H_temp=np.exp(1j*k*z*np.sqrt(1-(2*np.pi/k*FX)**2+(2*np.pi/k*FY)**2)) #trans func of RS
  ## ask matan how to - change range only to fx^2+fy^2 < 1/lambda^2
  rs_range = FX**2+FY**2
  rs_range_mask = np.where(rs_range < (1/lambda_in**2) ,1,0)
  H = H_temp * rs_range_mask ## only if in range, otherwise 0
  H=np.fft.fftshift(H) #shift trans func
  return torch.tensor(H, dtype=torch.cfloat, device=torch.device('cuda:0')) # ask Matan

def rect(x):
  #rectangle function
  return abs(x)<=1/2

def pearson_coef(output,target,device=torch.device('cuda:0')):
    eps = 10 ** -7
    x = output
    y = target

    vx = x - torch.mean(x)
    #vx = vx / vx.max() # normalization does not change pearson
    vy = y - torch.mean(y)
    #vy = vy / vy.max() # normalization does not change pearson
    res = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + eps)
    return res


# # function to calculate accuracy of the model
# def calculate_accuracy(model, dataloader, device=torch.device('cuda:0')):
#     model.eval() # put in evaluation mode
#     total_correct = 0
#     total_images = 0
#     confusion_matrix = np.zeros([10,10], int)
#     with torch.no_grad():
#         for data in dataloader:
#             images, labels = data
#             images = images.to(device)
#             labels = labels.to(device)
#             outputs = model(images).to(device) ##maby not need here to device
#             _, predicted = torch.max(outputs.data, 1).to(device)
#             total_images += labels.size(0)
#             total_correct += (predicted == labels).sum().item()
#             for i, l in enumerate(labels):
#                 confusion_matrix[l.item(), predicted[i].item()] += 1
#
#     model_accuracy = total_correct / total_images * 100
#     return model_accuracy, confusion_matrix

class customLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(customLoss, self).__init__()
        self.mse = nn.MSELoss() # ask Matan

    def forward(self, inputs, outputs, labels, target_label, G_k):
      # hyper parameters:
        a1 = 1
        a2 = 3
        b1 = 6
        b2 = 3
        b3 = 2
        loss_p = 0
        loss_n = 0
        #pearson = PearsonCorrCoef()
        #G_k_flat = torch.reshape(G_k, (-1,))
        for i in range(inputs.shape[0]):
          #input_flat = torch.reshape(inputs[i][0], (-1,))
          #output_flat = torch.reshape(outputs[i][0], (-1,))
          norm_output = outputs[i][0] / outputs[i][0].max()
          #norm_input = inputs[i][0] / inputs[i][0].max()
          #norm_G_k = G_k / G_k.max()
          if (labels[i] == target_label): # calculate the +loss
            #print('got target')
            # calculate NMSE
            nmse = self.mse(inputs[i][0], norm_output)
            # calculate PCC
            # pcc = pearson_coef(outputs[i][0], inputs[i][0])
            pcc = pearson_coef(outputs[i][0], inputs[i][0])
            #pcc = pearson(output_flat, input_flat)
            # calculate loss_p
            loss_p += a1 * nmse + a2 * (1-pcc)
          else: # calculate the -loss
            #print('got not target')
            # calculate pcc of input and output
            #pcc1 = pearson(output_flat, input_flat)
            #pcc1 = pearson_coef(outputs[i][0], inputs[i][0])
            pcc1 = pearson_coef(outputs[i][0], inputs[i][0])

            # calculate pcc of G_k and output
            #pcc2 = pearson(output_flat, G_k_flat)
            # pcc2 = pearson_coef(outputs[i][0], G_k)
            pcc2 = pearson_coef(outputs[i][0], G_k[0])
            # calculate pcc of output_shift and output

            #oShift = torchvision.transforms.functional.affine(outputs[i], angle=0,scale=1, shear=0,translate=(5,5))
            oShift = torch.roll(outputs[i][0], shifts=(5, 5), dims=(0, 1))
            # for tensorboard
            oshift_img = torchvision.utils.make_grid(oShift)
            # writer_2.add_image('oshift_img', oshift_img)

            pcc3 = pearson_coef(outputs[i][0], oShift)
            # calculate loss_n
            loss_n += (b1 * torch.abs(pcc1)) + (b2 * torch.abs(pcc2)) + b3 * pcc3

        return (loss_p + loss_n)/inputs.shape[0]



# def main():
#     # this is the main function;
#     # it loads the MNIST data
#     # the model is an optical dnn simulation according to "image or not to image" paper
#     # trains the model
#     # evaluates the model
#
#     # hyper parameters to be decided from paper:
#     target_label = 2
#     epochs = 1000
#     batch_size_train = 60
#     batch_size_test = 1000
#     learning_rate = 0.03
#     momentum = 0.5
#     log_interval = 10
#     layer_size = 120
#
#     random_seed = 1
#     torch.backends.cudnn.enabled = False
#     torch.manual_seed(random_seed)
#
#     # define optic simulation parameters - tbd from article
#     L1 = 0.5  # side length
#     # M = 250  # number of samples
#     # dx1 = L1 / M  # src sample interval
#     # x1 = np.arange(-L1 / 2, L1 / 2, dx1)  # src coords
#     # y1 = x1
#
#     lambda_in = 0.75 * (10 ** -3)  # wavelength in m
#     k = 2 * np.pi / lambda_in  # wavenumber
#     # w = 0.051  # source half width (m)
#     z = 20 * (10 ** -3)  # propagation dist (m)
#     # X1, Y1 = np.meshgrid(x1, y1)
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     ## RS approximation -
#     H = RS_estimation(layer_size, L1, lambda_in, z)  ##RS approximation
#
#     # each diffractive layer contains 120x120 diffractive pixels - ask Matan
#     # each pixel/neuron size is 0.4mm - ask Matan
#     # data augmentations (for each image)
#     # linearly upscale 28x28 to 90x90
#     # random transformations:
#     # rotation by an angle within [-10,10]
#     # random scaling by a factor within [0.9,1.1]
#     # random shift in each lateral direction by an amount of [-2.13lambda,2.13lambda] - ask Matan (horizontal shift is randomly
#     # sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.)
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.Resize([90,90]),
#         torchvision.transforms.RandomRotation([-10, 10]),
#         torchvision.transforms.RandomAffine(degrees=[0.9, 1.1],translate=(2.13 * lambda_in, 2.13 * lambda_in)),
#         torchvision.transforms.Pad([15,15]),
#         torchvision.transforms.ToTensor(),
#     ])
#
#     # 48000 train , 12000 validation, 10000 test - ask Matan
#     # data loader sampler to not sample all MNIST
#     train_set = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
#     train_samples = torch.randperm(train_set.data.shape[0])[:5000]
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,sampler=train_samples)
#
#     # train_loader = torch.utils.data.DataLoader( # data loader sampler to not sample all MNIST
#     #     torchvision.datasets.MNIST('../data', train=True, download=True,
#     #                                transform=transform), batch_size=batch_size_train, shuffle=True)
#
#     test_loader = torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('../data', train=False, download=True,
#                                    transform=transform), batch_size=batch_size_test, shuffle=True)
#     print('after download')
#     examples = enumerate(test_loader)
#     batch_idx, (example_data, example_targets) = next(examples)
#
#     # get a list of target class from data loader to choose random target digit G_k
#     dataset = train_loader.dataset
#     target_data_set = list(filter(lambda i: i[1] == target_label, dataset))
#     n_target_samples = len(target_data_set)
#
#     fig = plt.figure()
#     for i in range(6):
#         plt.subplot(2, 3, i + 1)
#         plt.tight_layout()
#         plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#         plt.title("Ground Truth: {}".format(example_targets[i]))
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
#
#     # time to train our model
#     # device - cpu or gpu?
#     #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## put it up
#
#     # loss criterion
#     criterion = customLoss()
#
#     # build our model and send it to the device
#     model = OpticModel(H, layer_size).to(device)  # no need for parameters as we already defined them in the class
#
#     # optimizer - SGD, Adam, RMSProp...
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     # training loop
#     for epoch in range(1, epochs + 1):
#         model.train()  # put in training mode
#         running_loss = 0.0
#         epoch_time = time.time()
#         for i, data in enumerate(train_loader, 0):
#             # get the inputs
#             inputs, labels = data
#
#             # randomly choose G_k from target_label in training set: (for each batch. using this in loss calculations)
#             # Get a random sample
#             # Get a random sample
#             random_index = int(np.random.random() * n_target_samples)
#             G_k = target_data_set[random_index][0].to(device)
#
#             # send them to device
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             # augmentation with `kornia` happens here inputs = aug_list(inputs)
#
#             # forward + backward + optimize
#             outputs = model.forward(inputs)  # forward pass
#             loss = criterion(inputs, outputs, labels, target_label,
#                              G_k)  # calculate the loss: +loss for target label, anf -loss for all other. choose G_k randomly from inputs
#             # always the same 3 steps
#             optimizer.zero_grad()  # zero the parameter gradients
#             loss.backward()  # backpropagation
#             optimizer.step()  # update parameters
#
#             # print statistics
#             running_loss += loss.data.item()
#
#         # Normalizing the loss by the total number of train batches
#         running_loss /= len(train_loader)
#
#         # Calculate training/test set accuracy of the existing model
#         # train_accuracy, _ = calculate_accuracy(model, train_loader, device)
#         # test_accuracy, _ = calculate_accuracy(model, test_loader, device)
#
#         log = "Epoch: {} | Loss: {:.4f} | ".format(epoch, running_loss)
#         # with f-strings
#         # log = f"Epoch: {epoch} | Loss: {running_loss:.4f} | Training accuracy: {train_accuracy:.3f}% | Test accuracy: {test_accuracy:.3f}% |"
#         epoch_time = time.time() - epoch_time
#         log += "Epoch Time: {:.2f} secs".format(epoch_time)
#         # with f-strings
#         # log += f"Epoch Time: {epoch_time:.2f} secs"
#         print(log)
#
#         # save model
#         if epoch % 20 == 0:
#             print('==> Saving model ...')
#             state = {
#                 'net': model.state_dict(),
#                 'epoch': epoch,
#             }
#             if not os.path.isdir('checkpoints'):
#                 os.mkdir('checkpoints')
#             torch.save(state, './checkpoints/mnist_optic.pth')
#
#     print('==> Finished Training ...')
#
#
# if __name__ == "__main__":
#         main()
