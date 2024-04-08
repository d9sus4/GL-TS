import numpy as np

def merge_topK(prob,topK=2):
    tmp = np.copy(prob)
    ranked_prob = -1*np.sort(-1*prob,axis=-1)[:,[topK]]
    tmp = tmp - ranked_prob
    merged_prob = np.copy(prob)
    merged_prob[tmp<=0] = -1
    merged_prob[merged_prob>0]=1
    return merged_prob

def adapt_merge_top(prob,gt=None):
    # merged_prob = -1*np.ones(prob.shape)
    sorted_prob = np.sort(prob,axis=-1)
    cumu_prob = np.cumsum(sorted_prob,axis=-1)
    diff = sorted_prob[:,1:] - cumu_prob[:,:-1]

    diff = diff[:,-2:]

    indicator = np.sum((sorted_prob[:,[-1]]<0.9)*(diff<-0.01), axis=-1)

    topK = 2*np.ones(prob.shape[0]).astype('int64')
    topK[indicator>0]=3

    conf_class_list = 10*np.ones((prob.shape[0],3))
    sorted_class = np.argsort(-1*prob,axis=-1)[:,:3]

    conf_class_list[topK==2,:2] = sorted_class[topK==2,:2]
    conf_class_list[topK==3] = sorted_class[topK==3]

    conf_class_list = np.sort(conf_class_list,axis=-1).astype('int')

    tmp = np.min(np.abs(conf_class_list-gt.reshape(-1,1)),axis=-1)
    print('topK acc = {}'.format(np.sum(tmp==0)/tmp.size))

    return conf_class_list

def logical_AND_multi(log1,log2,*args):
    length = len(args)
    indicator = log1*log2
    for i in range(length):
        indicator*=args[i]
    return indicator

def logical_OR_multi(log1,log2,*args):
    length = len(args)
    indicator = log1+log2
    for i in range(length):
        indicator+=args[i]
    return indicator

def cal_top2_acc(y_prob, y_gt, top=2):
    te_argsort = np.argsort(-1 * y_prob, axis=1)

    # top2 acc
    te_top2_pred = te_argsort[:, :top]
    tmp = te_top2_pred - y_gt.reshape(-1, 1)
    te_top2_acc = np.sum(np.min(np.abs(tmp), axis=1) == 0) / y_gt.size

    return te_top2_acc

def sort_pair_list(candid):
    # candid is the argsort of the 10 probabilities
    pair_list_sorted = []
    pair_num = []
    for PAIR0 in range(10):
        for PAIR1 in range(10):
            if PAIR1 > PAIR0:
                pair_list_sorted.append([PAIR0, PAIR1])
                pair_num.append(np.sum(logical_AND_multi(candid[:, 0] == PAIR0, candid[:, 1] == PAIR1)))
    pair_num = np.array(pair_num).squeeze()
    pair_list_sorted = np.array(pair_list_sorted)
    sort_idx = np.argsort(-1 * pair_num)
    pair_list_sorted = pair_list_sorted[sort_idx]
    sorted_pair_num = pair_num[sort_idx]

    return pair_list_sorted, sorted_pair_num,sort_idx

def get_groups_TRAIN_2(tr_y, PAIR0, PAIR1):
    tr_chosen = logical_OR_multi(tr_y == PAIR0, tr_y == PAIR1).squeeze()
    ynew = np.copy(tr_y[tr_chosen])
    ynew[ynew == PAIR0] = 0
    ynew[ynew == PAIR1] = 1
    print(np.unique(ynew))

    return ynew, tr_chosen

def get_groups_TEST_2(y, y_top2, PAIR0, PAIR1):
    '''top2 in pair0 pair1, and ground truth also pair0 pair1'''
    # y_top2 = np.tile(y_top2.T, len(AUG_list)).T
    chosen = logical_AND_multi(logical_AND_multi(y_top2[:, 0] == PAIR0, y_top2[:, 1] == PAIR1).squeeze(),
                               logical_OR_multi(y == PAIR0, y == PAIR1).squeeze())
    ynew = np.copy(y[chosen])
    ynew[ynew == PAIR0] = 0
    ynew[ynew == PAIR1] = 1
    print(np.unique(ynew))
    print('test for {} vs {} = {}'.format(PAIR0, PAIR1, np.sum(chosen)))
    return ynew, chosen


def get_groups_TRAIN_3(tr_y, PAIR0, PAIR1,PAIR2):
    tr_chosen = logical_OR_multi(tr_y == PAIR0, tr_y == PAIR1, tr_y == PAIR2).squeeze()
    ynew = np.copy(tr_y[tr_chosen])
    ynew[ynew == PAIR0] = 0
    ynew[ynew == PAIR1] = 1
    ynew[ynew == PAIR2] = 2

    print(np.unique(ynew))

    return ynew, tr_chosen

def get_groups_TEST_3(y, y_top3, PAIR0, PAIR1,PAIR2):
    '''top2 in pair0 pair1, and ground truth also pair0 pair1'''
    # y_top2 = np.tile(y_top2.T, len(AUG_list)).T
    chosen = logical_AND_multi(logical_AND_multi(y_top3[:, 0] == PAIR0, y_top3[:, 1] == PAIR1, y_top3[:, 2] == PAIR2).squeeze(),
                               logical_OR_multi(y == PAIR0, y == PAIR1, y == PAIR2).squeeze())
    ynew = np.copy(y[chosen])
    ynew[ynew == PAIR0] = 0
    ynew[ynew == PAIR1] = 1
    ynew[ynew == PAIR2] = 2

    print(np.unique(ynew))
    print('test for {} vs {} = {}'.format(PAIR0, PAIR1, np.sum(chosen)))
    return ynew, chosen
