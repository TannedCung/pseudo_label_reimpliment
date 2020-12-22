import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class earlyRegu(nn.Module):
    def __init__(self, num_classes):
        super(earlyRegu, self).__init__()
        # self.pre_hc = torch.zeros(1, num_classes)
        self.Pc = 1/num_classes*torch.ones(1, num_classes)
        self.init = True
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output):
        output = self.softmax(output)
        if self.init:
            pre_sum = torch.sum(output, dim=0)/output.shape[0]
            self.pre_hc = pre_sum.data
            self.init = False
            return torch.zeros(1)
        else:
            loss = torch.sum(self.Pc*torch.log10(self.Pc/self.pre_hc))
            pre_sum = torch.sum(output, dim=0)/output.shape[0]
            self.pre_hc = pre_sum.data
            return loss

class EntropyRegu(nn.Module):
    def __init__(self):
        super(EntropyRegu, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output):
        output = self.softmax(output)
        loss = torch.sum(output*torch.log10(output))/output.shape[0]
        return loss