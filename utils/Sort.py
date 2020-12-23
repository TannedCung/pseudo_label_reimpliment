import torch
from dataset.dataloader import NO_LABEL


def sort_XbyY(X, Y):
    Y, idxs = Y.sort()
    X_p = X.clone()
    for i, idx in enumerate(idxs):
        X_p[i] = X[idx]
    
    return X_p, Y

def split_labeled_unlabeled(X, Y):
    mask = Y.eq(NO_LABEL)
    X_mask = X.clone()
    for i, v in enumerate(mask):
        X_mask[i] = v.float()

    X_labeled = X_mask.ne(1)*X
    X_unlabeled = X_mask*X
    # Y_labeled = mask.ne(1)*Y
    Y_unlabeled = mask*Y

    X_labeled = remove_zeros(X_labeled, num_dim=4)
    X_unlabeled = remove_zeros(X_unlabeled, num_dim=4)
    Y_labeled = remove_zeros(Y, num_dim=1, remove=-1)
    Y_unlabeled = remove_zeros(Y_unlabeled, num_dim=1)

    return X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, mask

def remove_zeros(data, num_dim, remove=0):
    if num_dim == 4:
        return data[torch.abs(data).sum(dim=3).sum(dim=2).sum(dim=1)!=remove]
    elif num_dim == 3:
        return data[torch.abs(data).sum(dim=2).sum(dim=1)!=remove]
    elif num_dim == 2:
        return data[torch.abs(data).sum(dim=1)!=remove]
    elif num_dim == 1:
        return data[data!=remove]


def unlabeled_weight(epoch, T1=50, T2=600, af=0.3):
    alpha = 0.0
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha
        

