import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=1):
        super(LabelSmoothingLoss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        output = output.log_softmax(dim=-1)
        true_dist = output.clone()
        true_dist.fill_(self.smoothing / (self.tgt_vocab_size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0
        mask = torch.nonzero(target.data != self.ignore_index)
        return self.criterion(output, true_dist) / mask.size(0)
