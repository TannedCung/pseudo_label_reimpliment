import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from dataset.dataloader import * 
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from dataset.dataloader import *
from losses_regularization.mixup_on_data import *
from losses_regularization.losses import *
from losses_regularization.regularizations import *
from networks.metrics import *
from networks.models import *
import time
from tqdm import tqdm
from torch import nn

from dataset.dataloader import NO_LABEL

# Problem may occur: Mixer mixes Y by indexes which may lead to appropriate axis

BATCH_SIZE = 8
LABELED_RATIO = 0.2
LABELED_BATCH_SIZE = int(BATCH_SIZE*LABELED_RATIO)
DATA_DIR = 'data'
NUM_WORKERS = 2
NUM_CLASSES = 2 
ALL_CLS_REGU_WEIGHT = 0.8
ENTROPY_REGU_WEIGHT = 0.4
START_EPOCH = 0 
END_EPOCH = 1000
ALPHA = 4
MODEL_SAVE = './checkpoints/model.pth'

## ----- Create model -------
Net = MobileFaceNetUltraLite(embedding_size=NUM_CLASSES)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Net.to(device)
print(device)

## ----- Losses -------------------
criterion_cls = CategoricalCrossEntropy()
all_cls_regu = earlyRegu(num_classes=NUM_CLASSES)
entropy_regu = EntropyRegu()

## ----- Dataloader --------------------

trainset = maskDataset(data_dir=DATA_DIR, batch_size=BATCH_SIZE, labeled_percents=LABELED_RATIO)
labeled_idxs = trainset.labeled_idxs
unlabeled_idxs = trainset.unlabeled_idxs
batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, BATCH_SIZE, LABELED_BATCH_SIZE)
data = torch.utils.data.DataLoader(trainset,
                                    batch_sampler=batch_sampler,
                                    # num_workers=NUM_WORKERS,
                                    pin_memory=True)

## ----- Train --------------------

Net.train()
train_loss = 0
correct = 0
total = 0
running_loss = 0
start = time.time()
Data_Mixer = Mixer(alpha=ALPHA)
opt = optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)
best_loss = 1000.0
sm = nn.Softmax(dim=1)

for epoch in range(START_EPOCH, END_EPOCH):
    train_loss = 0
    start = time.time() 
    for i,d in enumerate(data):
        [X, Y] = d[0].to(device), d[1].to(device)
        Y_oh = onehot(Y, num_classes=NUM_CLASSES).to(device)
        opt.zero_grad()
        # ------ First inference to get soft labels --------
        p_Y = Net(X)
        pseudo_Y = sm(p_Y)
        # Merge pseudo labels and Ground Truth labels
        Y_oh = Y_oh*(Y_oh.ne(NO_LABEL)) + Y_oh.eq(NO_LABEL)*pseudo_Y
        # mixup_data
        mixed_X, Y_p, Y_q , lamda = Data_Mixer.mixup_data(X, Y_oh)
        # ------ Second inference means the real train --------
        opt.zero_grad()

        output = Net(mixed_X)
        cls_loss = Data_Mixer.mixup_loss(criterion=criterion_cls, output=output, Y_p=Y_p, Y_q=Y_q, lamda=lamda)
        local_loss = cls_loss + ENTROPY_REGU_WEIGHT*entropy_regu(output=output) + ALL_CLS_REGU_WEIGHT*all_cls_regu(output).to(device)
        loss = local_loss
        loss.backward()
        opt.step()
        running_loss += loss.item()
        train_loss += loss.item()*len(Y)
        if i % 10 == 9:
            print("[{}, {}], loss {:.10} in {:.5}s".format(epoch+1, i+1, running_loss/10, time.time()-start))
            # save_progress(state="   BATCH [{}, {}]".format(i-10, i), epoch= epoch+1, train_loss=running_loss/10, train_acc=100*correct/total)
            start = time.time()     
            running_loss = 0.0
    print ("====== Epoch {} Loss: {:.5}======".format(epoch+1, train_loss/len(data.sampler)))

    if best_loss >= train_loss/len(data.sampler):
        torch.save(Net, MODEL_SAVE)
        # torch.save(metric_fc.weight, HEAD_PTH)
        print("model saved to {}".format(MODEL_SAVE))
        best_loss = train_loss/len(data.sampler)
        # save_progress(state="SAVED   ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=best)

    else:
        print("model not saved as best_loss <= train_loss, current best : {}".format(best_loss))
        # save_progress(state="FAIL    ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=100*correct/total)
