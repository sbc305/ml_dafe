import os
from pathlib import Path

import wandb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import textwrap
from tqdm.notebook import tnrange

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import ProtoNet, load_protonet_conv
from load_data import read_images, extract_sample


def wrap_label(label, width=60):
    return '\n'.join(textwrap.wrap(label, width=width)).replace('/', '/\n')

def save_test_inference_table(sample, model, save_path="./test_inference.png", class_labels=None):
    with torch.no_grad():
        loss, out = model.set_forward_loss(sample)

    images = sample['images'].cpu()
    n_way, n_total, c, h, w = images.shape
    n_support = sample['n_support']
    n_query = sample['n_query']
    y_hat = out['y_hat'].cpu().numpy()

    query_images = images[:, n_support:, :, :, :]

    total_queries = n_way * n_query

    fig = plt.figure(figsize=(6, total_queries * 1.5))
    spec = gridspec.GridSpec(total_queries, 3, width_ratios=[1.5, 1, 1])
    fig.suptitle("Query Predictions", fontsize=16)

    row = 0
    for i in range(n_way):
        for j in range(n_query):
            image = query_images[i, j].permute(1, 2, 0).numpy() / 255.0
            pred_label = y_hat[i * n_query + j]
            true_label = i

            pred_name = wrap_label(class_labels[pred_label] if class_labels is not None else f"class {pred_label}")
            true_name = wrap_label(class_labels[true_label] if class_labels is not None else f"class {true_label}")

            ax0 = fig.add_subplot(spec[row, 0])
            ax0.imshow(image)
            ax0.axis("off")
            ax0.set_title("Query Image", fontsize=10)

            ax1 = fig.add_subplot(spec[row, 1])
            ax1.axis("off")
            ax1.text(0.5, 0.5, pred_name, fontsize=8, ha='center', va='center',
                     color='green' if pred_label == true_label else 'red')
            ax1.set_title("Predicted", fontsize=10)

            ax2 = fig.add_subplot(spec[row, 2])
            ax2.axis("off")
            ax2.text(0.5, 0.5, true_name, fontsize=8, ha='center', va='center')
            ax2.set_title("Ground Truth", fontsize=10)

            row += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved test inference visualization to: {save_path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
    return model

n_way = 5
n_support = 5
n_query = 5

model = load_model(load_protonet_conv(x_dim=(3, 28, 28), hid_dim=64, z_dim=64), "checkpoints/model5.pth")
optimizer = optim.Adam(model.parameters(), lr=0.001)
test_x, test_y = read_images('images_evaluation')

test_episode = 1000

sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
save_test_inference_table(sample, model, class_labels=np.unique(test_y))
