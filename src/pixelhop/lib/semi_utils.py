import numpy as np

def randsamp_balanced(y, num = 10, shuffle=False):
    EACH_ = num//np.unique(y).size
    idx = []
    for c in range(np.unique(y).size):
        tmp = np.argwhere(y==c).squeeze()
        idx.append(np.random.permutation(tmp)[:EACH_])
    idx = np.concatenate(idx).reshape(-1)
    if shuffle == True:
        idx = np.random.permutation(idx)
    else:
        idx = np.sort(idx)
    return idx

def randsamp_balanced_eachnum(y, eachnum = 10, shuffle=False):
    EACH_ = eachnum
    idx = []
    for c in range(np.unique(y).size):
        tmp = np.argwhere(y==c).squeeze()
        idx.append(np.random.permutation(tmp)[:EACH_])
    idx = np.concatenate(idx).reshape(-1)
    if shuffle == True:
        idx = np.random.permutation(idx)
    else:
        idx = np.sort(idx)
    return idx
