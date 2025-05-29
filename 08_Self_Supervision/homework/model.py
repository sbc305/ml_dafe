import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(x_dim[0], hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(hid_dim, z_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.cpu()

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat (predict)
        """
        sample_images = sample['images']
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']
        
        # pass your code
        
        support_images = sample_images[:, :n_support, :, :, :].reshape(n_way * n_support, 3, 28, 28)
        query_images = sample_images[:, n_support:, :, :, :].reshape(n_way * n_query, 3, 28, 28)

        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        support_embeddings = support_embeddings.view(n_way, n_support, -1)
        prototypes = support_embeddings.mean(dim=1)

        dists = torch.cdist(query_embeddings.unsqueeze(1), prototypes.unsqueeze(0))
        dists = dists.squeeze(1)

        log_p_y = F.log_softmax(-dists, dim=1)

        target_inds = torch.arange(n_way).repeat_interleave(n_query).long().cpu()

        loss_val = F.nll_loss(log_p_y, target_inds)

        y_hat = log_p_y.argmax(dim=1)
        acc_val = (y_hat == target_inds).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat
            }

def load_protonet_conv(x_dim=(3, 28, 28), hid_dim=64, z_dim=64):
    """
    Loads the prototypical network model
    Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (Class ProtoNet)
    """
    encoder = Encoder(x_dim=x_dim, hid_dim=hid_dim, z_dim=z_dim)
    return ProtoNet(encoder)
