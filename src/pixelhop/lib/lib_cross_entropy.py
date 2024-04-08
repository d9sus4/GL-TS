import numpy as np

def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    num_cls = targets.shape[1]
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce/np.log(num_cls)

def cal_entropy(prob,num_cls=10):
    prob_tmp = np.copy(prob)
    prob_tmp[prob_tmp==0] = 1
    tmp = np.sum(-1*prob*np.log(prob_tmp),axis=-1)
    return tmp/np.log(num_cls)

def class_distribution(y, num_cls=10):
    distribution = np.zeros(num_cls)
    for c in range(num_cls):
        distribution[c] = y.tolist().count(c)
    distribution = distribution/np.sum(distribution)
    return distribution

def cal_weighted_CE(X, y, bound, num_cls=10):
    classwise = np.zeros(num_cls)
    for c in range(num_cls):
        classwise[c] = np.sum(y==c)

    # only two bins
    left_y = y[X<bound]
    right_y = y[X>=bound]
    left_y_onehot = np.eye(num_cls)[left_y]
    right_y_onehot = np.eye(num_cls)[right_y]

    left_num = left_y.size
    right_num = right_y.size

    left_pred = np.repeat(left_num*class_distribution(left_y, num_cls=num_cls).reshape(1,-1),left_num,axis=0)
    right_pred = np.repeat(right_num*class_distribution(right_y, num_cls=num_cls).reshape(1,-1),right_num,axis=0)

    left_pred = left_pred/classwise.reshape(1,-1)
    right_pred = right_pred/classwise.reshape(1,-1)

    left_pred = left_pred/np.sum(left_pred,axis=-1,keepdims=1)
    right_pred = right_pred/np.sum(right_pred,axis=-1,keepdims=1)

    ce = np.array([cross_entropy(left_pred, left_y_onehot),cross_entropy(right_pred, right_y_onehot)]).reshape(-1,1)
    num = np.array([left_num, right_num]).reshape(1,-1)
    num = num/np.sum(num,keepdims=1)
    wCE = num @ ce
    return wCE