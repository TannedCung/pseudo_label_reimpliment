import torch
from torch import nn
from dataset.dataloader import NO_LABEL
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F

class CategoricalCrossEntropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        logits = self.softmax(output)
        loss = torch.sum(-target*torch.log10(logits))
        return loss/target.shape[0]

def onehot(labels, num_classes):
    placehoder = torch.zeros([labels.shape[0], num_classes])
    for i, l in enumerate(labels):
        if l != NO_LABEL:
            placehoder[i][l] = 1.0
        else:
            placehoder[i] = -1.0

    return placehoder

class MyCrossEntropyLoss(_WeightedLoss):
    def init(self, weight=None, reduction='mean'):
        super().init(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss

class CategoricalCrossEntropy2(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy2, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        logits = self.softmax(output)
        loss = torch.sum(-target*torch.log(logits))
        return loss/target.shape[0]