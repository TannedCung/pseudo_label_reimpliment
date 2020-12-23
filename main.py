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
from utils.Sort import *
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

# Problem may occur: Mixer mixes Y by indexes which may lead to INappropriate axis
WARM_UP = 30
BATCH_SIZE = 128
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
warm_up_criterion = nn.CrossEntropyLoss()
criterion_cls = CategoricalCrossEntropy()
all_cls_regu = earlyRegu(num_classes=NUM_CLASSES)
entropy_regu = EntropyRegu()

## ----- Dataloader --------------------

trainset = maskDataset(data_dir=DATA_DIR, labeled_percents=LABELED_RATIO)
labeled_idxs = trainset.labeled_idxs
unlabeled_idxs = trainset.unlabeled_idxs
batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, BATCH_SIZE, LABELED_BATCH_SIZE)
data = torch.utils.data.DataLoader(trainset,
                                    batch_sampler=batch_sampler,
                                    # num_workers=NUM_WORKERS,
                                    pin_memory=True)
# ----- Warm up for a few epochs --------
warm_up_set = WarmUpDataset(data_dir=DATA_DIR)
warm_up_data = torch.utils.data.DataLoader(warm_up_set,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
# ------ Validate ---------
valid_set = WarmUpDataset(data_dir=DATA_DIR, subfolder='test')
valid_data = torch.utils.data.DataLoader(warm_up_set,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
def validate(Net, dataloader):
    Net.eval()
    valid_correct = 0
    for i, d in enumerate(valid):
        [X, Y] = d[0].to(device), d[1].to(device)
        out = Net(X)
        valid_correct += torch.sum(out.argmax(dim=1).float().eq(Y)).item()
    print ("====== Test on testset acc: {:.5}% ======".format(100*valid_correct/len(valid_data.sampler)))
    return 100*valid_correct/len(valid_data.sampler)

Net.train()
opt = optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)
warmup_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6, 12], gamma=0.1)
for epoch in range(WARM_UP):
    running_loss = 0
    warm_loss = 0
    warm_acc = 0
    start = time.time()
    warm_total = 0
    warm_correct = 0
    for i,d in enumerate(warm_up_data):
        [X, Y] = d[0].to(device), d[1].to(device)
    
        opt.zero_grad()
        out = Net(X)
        loss = warm_up_criterion(out, Y)
        loss.backward()
        opt.step()
        warm_loss += loss.data*BATCH_SIZE

        warm_correct += torch.sum(out.argmax(dim=1).float().eq(Y)).item()
    print ("====== Warm up epoch {} Loss: {:.5}, acc: {:.5}% ======".format(epoch+1, warm_loss/len(warm_up_data.sampler), 100*warm_correct/len(warm_up_data.sampler)))

## ----- Real Train --------------------

Net.train()
train_loss = 0
correct = 0
total = 0
running_loss = 0
start = time.time()
Data_Mixer = Mixer(alpha=ALPHA)
opt = optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)
best = 0
sm = nn.Softmax(dim=1)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[6, 48, 96, 196, 320], gamma=0.5)
for epoch in range(START_EPOCH, END_EPOCH):
    Net.train()
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
        # Y_oh = Y_oh*(Y_oh.ne(NO_LABEL)) + Y_oh.eq(NO_LABEL)*pseudo_Y
        X_l, Y_l, X_ul, Y_ul, _ = split_labeled_unlabeled(X, Y)
        # mixup_data
        mixed_X, Y_p, Y_q , lamda = Data_Mixer.mixup_data(X_l, onehot(Y_l,num_classes=NUM_CLASSES))
        # ------ Second inference means the real train --------
        opt.zero_grad()
        # labeled phase
        output_l = Net(mixed_X)
        labeled_cls_loss = Data_Mixer.mixup_loss(criterion=criterion_cls, output=output_l, Y_p=Y_p, Y_q=Y_q, lamda=lamda)
        labeled_loss = labeled_cls_loss + ENTROPY_REGU_WEIGHT*entropy_regu(output=output_l) + ALL_CLS_REGU_WEIGHT*all_cls_regu(output_l).to(device)
        # unlabeled phase
        output_ul = Net(X_ul)
        Y_ul = 0*(Y_oh.ne(NO_LABEL)) + Y_oh.eq(NO_LABEL)*pseudo_Y
        Y_ul = remove_zeros(Y_ul, num_dim=2)
        unlabeled_cls_loss = criterion_cls(output_ul, Y_ul)
        unlabeled_loss = unlabeled_cls_loss + ENTROPY_REGU_WEIGHT*entropy_regu(output=output_ul) + ALL_CLS_REGU_WEIGHT*all_cls_regu(output_ul).to(device)
        # sum loss
        loss = labeled_loss + unlabeled_weight(epoch=epoch)*unlabeled_loss
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

    acc = validate(Net, valid_data)
    if best< acc:
        torch.save(Net, MODEL_SAVE)
        # torch.save(metric_fc.weight, HEAD_PTH)
        print("model saved to {}".format(MODEL_SAVE))
        best = acc 
        # save_progress(state="SAVED   ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=best)

    else:
        print("model not saved as best > acc, current best : {}".format(best))
       # save_progress(state="FAIL    ", epoch= epoch+1, train_loss=train_loss/len(data.sampler), train_acc=100*correct/total)
    