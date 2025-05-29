import os
from pathlib import Path

import wandb

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import ProtoNet, load_protonet_conv
from load_data import read_images, extract_sample



def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size, save=False):
    """
    Trains the protonet
    Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0
    stop = False

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in trange(epoch_size, desc="Epoch {:d} train".format(epoch + 1)):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
            loss.backward()
            optimizer.step()
            log_dict = {
                "Train Loss": output['loss'],
                "Accuracy": output['acc'],
                "Episode": epoch*epoch_size + episode,
            }
            wandb.log(log_dict)
        
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
        epoch += 1
        scheduler.step()

        if save:
            save_path = f"./checkpoints/model{epoch}.pth"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)
            print(f"Model saved to {save_path}")

n_way = 60
n_support = 5
n_query = 5
epochs = 5
epoch_size = 2000
wandb.init(
    project="hw4",
    config={
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
        "epochs": epochs,
        "epoch_size": epoch_size
    }
)

model = load_protonet_conv(
    x_dim=(3, 28, 28),
    hid_dim=64,
    z_dim=64,
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_x, train_y = read_images('images_background')

train(model, optimizer, train_x, train_y, n_way, n_support, n_query, epochs, epoch_size, True)

# n_way = 5
# n_support = 5
# n_query = 5

# test_x = testx
# test_y = testy

# test_episode = 1000

# test(model, test_x, test_y, n_way, n_support, n_query, test_episode)