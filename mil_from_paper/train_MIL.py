from __future__ import print_function, division
import os
import glob
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from get_512 import get_512_batch
from theirnet import TheirNet
from patch_data import PatchMethod
from tensorboardX import SummaryWriter

# writer = SummaryWriter('runs/MILonCPCTRreg1e_3')
# writer = SummaryWriter('runs/MILonCPCTR_drop0.7_reg3e_3')
# writer = SummaryWriter('runs/MILonCPCTR_drop0.6_reg4e_3')
writer = SummaryWriter('runs/MILonCPCTR_drop0.0_reg0.1_1')

data = PatchMethod(root = '/home/intern_swagatam/CPCTR/train')
val_data =PatchMethod(root = '/home/intern_swagatam/CPCTR/test', mode = 'test')

# data = PatchMethod(root = '/home/intern_swagatam/cpctr_25th_june/CPCTR_4/check')
# val_data =PatchMethod(root = '/home/intern_swagatam/cpctr_25th_june/CPCTR_4/check', mode = 'test')


trainloader = torch.utils.data.DataLoader(data, shuffle = True, num_workers = 6, batch_size = 1)
valloader = torch.utils.data.DataLoader(val_data, shuffle = True, num_workers = 6, batch_size = 1)

print(f'Training on {len(trainloader)} samples')
print(f'Validating on {len(valloader)} samples')

vgg_model = models.vgg16(pretrained = True)
vgg_model = vgg_model.cuda()
model_features = vgg_model.features

model = TheirNet()
model = model.cuda()

LR = 1e-5
# reg = 4e-3
reg = 0.1

optimizer = optim.Adam(model.parameters(), lr = LR, betas = (0.9, 0.999), weight_decay = reg)

num_epochs = 2000

print('Starting Training...')


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    count = 0
    train_loss_100 = 0.
    train_error_100 = 0.

    for i, (images, label) in enumerate(trainloader):
        images, label = images.cuda(), label.cuda()
        images = images.squeeze(0)

        with torch.no_grad():
            images = model_features(images)
            images = get_512_batch(images)
            pass

        optimizer.zero_grad()
        loss, _ = model.calculate_objective(images, label)
        train_loss += loss.item()
        train_loss_100 += loss.item()
        error, _ = model.calculate_classification_error(images, label)
        train_error += error
        train_error_100 += error
        # backward pass
        loss.backward()
        if (i+1) % 100 == 0:
            train_loss_100 /= 100
            train_error_100 /= 100
            print(f'After {i + 1} iterations, Loss = {train_loss_100}')
            writer.add_scalar('TrainingLossIteration', train_loss_100, i)
            writer.add_scalar('TrainingErrorIteration', train_error_100, i)
            train_loss_100 = 0.
            train_error_100 = 0.
        # count += 1
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(trainloader)
    train_error /= len(trainloader)
    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch+1, train_loss, train_error))
    writer.add_scalar('TrainingLossEpoch', train_loss, epoch)
    writer.add_scalar('TrainingErrorEpoch', train_error, epoch)
    torch.save(model, f'/home/intern_swagatam/MILonCPCTRcodes/mil_from_paper/saved_models/model{epoch+1}.pth')
    # if epoch % 5 == 0:
        # torch.save(model, f'/home/intern_swagatam/MILonCPCTRcodes/mil_from_paper/saved_models/model{epoch+1}.pth')


def test(epoch):
    model.eval()
    test_loss = 0.
    test_error = 0.

    with torch.no_grad():
        for i, (images, label) in enumerate(valloader):
            images, label = images.cuda(), label.cuda()
            images = images.squeeze(0)
            images = model_features(images)
            images = get_512_batch(images)

            loss, attention_weights = model.calculate_objective(images, label)
            test_loss += loss.item()
            error, predicted_label = model.calculate_classification_error(images, label)
            test_error += error
            if(i%10 == 0):
                print(f'Predicted Label: {predicted_label}, True Label: {label}, Error: {error}')



        test_error /= len(valloader)
        test_loss /= len(valloader)


    print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))
    writer.add_scalar('TestingLossEpoch', test_loss, epoch)
    writer.add_scalar('TestingErrorEpoch', test_error, epoch)


if __name__ == "__main__":
    print('Start Training')
    count = 0
    for epoch in range(2000):
        train(epoch)
        print('Start Testing')
        test(epoch)
