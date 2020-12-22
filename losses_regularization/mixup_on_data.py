import torch
from torch import nn

class Mixer():
    def __init__(self, alpha):
        if alpha >0 :
            self.alpha = alpha
        else:
            self.alpha = 1
        self.distribution = torch.distributions.beta.Beta(
                                torch.tensor([self.alpha*1.0]),
                                torch.tensor([self.alpha*1.0]))

    def mixup_data(self, X, y):
        lamda = self.distribution.sample()
        batch_size = X.size()[0]
        index = torch.randperm(batch_size)

        mixed_X = lamda*X + (1-lamda)*X[index, :]
        Y_p, Y_q = y, y[index]

        return mixed_X,  Y_p, Y_q, lamda

    def mixup_loss(self, criterion, output, Y_p, Y_q, lamda):
        return lamda*criterion(output, Y_p) + (1-lamda)*criterion(output, Y_q)

    