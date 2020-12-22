import torch
from torch import nn
from dataset.dataloader import NO_LABEL

class CategoricalCrossEntropy(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropy, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        logits = self.softmax(output)
        loss = torch.sum(-target*torch.log10(logits))
        return loss

def onehot(labels, num_classes):
    placehoder = torch.zeros([labels.shape[0], num_classes])
    for i, l in enumerate(labels):
        if l != NO_LABEL:
            placehoder[i][l] = 1.0
        else:
            placehoder[i] = -1.0

    return placehoder