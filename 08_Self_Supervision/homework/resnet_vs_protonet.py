import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import os
from model import ProtoNet, load_protonet_conv
from load_data import read_images, extract_sample

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

import numpy as np
from glob import glob
import textwrap

def load_model(model, path):
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    print("Model loaded successfully.")
    return model


def create_protonet_sample(datax, datay, n_way, n_support, n_query):
    unique_classes = torch.unique(datay)
    chosen_classes = unique_classes[torch.randperm(len(unique_classes))[:n_way]]

    sample = []
    query_info = []

    for cls in chosen_classes:
        cls_indices = torch.where(datay == cls)[0]
        cls_indices = cls_indices[torch.randperm(len(cls_indices))]
        selected = cls_indices[:(n_support + n_query)]
        cls_images = datax[selected]

        cls_images = interpolate(cls_images, size=(28, 28), mode='bilinear', align_corners=False)

        sample.append(cls_images)

        query_img = cls_images[n_support]
        query_info.append((query_img, cls.item()))

    sample = torch.stack(sample)

    return {
        'images': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    }, query_info

def wrap_label(label, width=60):
    return '\n'.join(textwrap.wrap(label, width=width)).replace('/', '/\n')

transform = transforms.Compose([transforms.ToTensor()])
cifar_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
datax = torch.stack([img for img, _ in cifar_test])
datay = torch.tensor([label for _, label in cifar_test])

n_way = 5
n_support = 5
n_query = 1
n_tasks = 10
save_path = "resnet_vs_protonet/"

proto_samples = []
query_images = []
query_labels = []

protonet = load_model(load_protonet_conv(x_dim=(3, 28, 28), hid_dim=64, z_dim=64), "./checkpoints/model5.pth").eval()

resnet = models.resnet18()
resnet = resnet.to('cpu')


num_classes = 10
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)
)
resnet = load_model(resnet, "./checkpoints/resnet/resnet_7.pth")

for _ in range(n_tasks):
    proto_sample, query_info = create_protonet_sample(datax, datay, n_way, n_support, n_query)
    proto_samples.append(proto_sample)
    
    for img, label in query_info:
        query_images.append(img)
        query_labels.append(label)

query_labels_ = query_labels
query_images = torch.stack(query_images)
query_images_resnet = interpolate(query_images, size=(32, 32), mode='bilinear', align_corners=False)
query_labels = torch.tensor(query_labels)

resnet.eval()
pred_resnet = []
with torch.no_grad():
    predictions = resnet(query_images_resnet.cpu())
    _, pred_resnet = torch.max(predictions.data, 1)

for i_, sample in enumerate(tqdm(proto_samples, desc="Processing samples")):
    with torch.no_grad():
        loss, out = protonet.set_forward_loss(sample)

    images = sample['images'].cpu()
    n_way, n_total, c, h, w = images.shape
    n_support = sample['n_support']
    n_query = sample['n_query']
    y_hat = out['y_hat'].cpu().numpy()

    query_images = images[:, n_support:, :, :, :]

    total_queries = n_way * n_query

    fig = plt.figure(figsize=(6, total_queries * 1.5))
    spec = gridspec.GridSpec(total_queries, 4, width_ratios=[1.5, 1, 1, 1])

    row = 0
    for i in range(n_way):
        for j in range(n_query):
            image = query_images[i, j].permute(1, 2, 0).numpy()
            pred_label = y_hat[i * n_query + j]
            true_label = i

            pred_name = wrap_label(f"class {pred_label}")
            true_name = wrap_label(f"class {true_label}")

            ax0 = fig.add_subplot(spec[row, 0])
            ax0.imshow(image)
            ax0.axis("off")
            ax0.set_title("Query Image", fontsize=10)

            ax1 = fig.add_subplot(spec[row, 1])
            ax1.axis("off")
            ax1.text(0.5, 0.5, pred_name, fontsize=8, ha='center', va='center',
                     color='green' if pred_label == true_label else 'red')
            ax1.set_title("ProtoNet", fontsize=10)

            ax2 = fig.add_subplot(spec[row, 2])
            ax2.axis('off')
            ax2.text(0.5, 0.5, str(pred_resnet[i]), fontsize=8, ha='center', va='center',
                     color='green' if pred_resnet[i] == true_label else 'red')
            ax2.set_title("ResNet18")

            ax3 = fig.add_subplot(spec[row, 3])
            ax3.axis("off")
            ax3.text(0.5, 0.5, true_name, fontsize=8, ha='center', va='center')
            ax3.set_title("Ground Truth", fontsize=10)

            row += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}{i_}.png")
    plt.close()

