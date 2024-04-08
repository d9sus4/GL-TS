'''
v2022.07.04
iphop2
'''

import numpy as np
import xgboost as xgb
import time
import os
import lib.feat_utils as FEAT

import pickle
import lib.layer as LAYER
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import lib.lib_global_saab as GS
# %% warning
import warnings
import argparse

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='iphop2')
    parser.add_argument('--dataset', default='mnist',help='dataset mnist/fashion')

    # parser.add_argument('--root', default='/feat/',help='feat_root')
    # parser.add_argument('--dataroot', default='/dataset/',help='data_root')
    # parser.add_argument('--saveroot', default='/result/',help='result_root')
    parser.add_argument('--root', default='/scratch1/yijingya/weak_sup/feat/0514_fullset__',help='feat_root')
    parser.add_argument('--dataroot', default='/project/jckuo_84/yijingya/dataset/',help='data_root')
    parser.add_argument('--saveroot', default='/scratch1/yijingya/weak_sup/results/iphop_combination_0704_',help='result_root')

    # no need to change
    parser.add_argument('--abs',default=1, type=int, help='absolute')
    parser.add_argument('--B',default=16, type=int, help='B for DFT')
    parser.add_argument('--knn_K',default=1, type=int, help='knn K')
    parser.add_argument('--NODC',default=0, type=int, help='sign')
    parser.add_argument('--POOL',default=2, type=int, help='pooling')

    # can be tuned
    parser.add_argument('--DIM1',default=400, type=int, help='DIM for FS')
    parser.add_argument('--DIM2',default=400, type=int, help='DIM for FS')

    parser.add_argument('--FS', default='DFT',help='feature selection method')
    parser.add_argument('--clf', default='xgb',help='XGBoost classifier')

    args = parser.parse_args()
    return args

args = parse_args()

HOPLIST = [0,1]
AUG_list = [0]
DIM_list = [args.DIM1, args.DIM2]

root = [args.root +  args.dataset + '/']

if args.FS == 'DFT':
    saveroot = args.saveroot + args.dataset + '_FS{}_B{}_dim{}_{}_clf{}/'.format(args.FS, args.B, args.DIM1, args.DIM2, args.clf)
else:
    saveroot = args.saveroot + args.dataset + '_FS{}_dim{}_{}_clf{}/'.format(args.FS, args.DIM1, args.DIM2, args.clf)

if not os.path.isdir(saveroot): os.makedirs(saveroot)



def get_feat_grouping_raw(hopidx=0, target=[3, 5], mode='tr', ROOT=None, chosenidx=None, aug_idx=0):
    if chosenidx is None:
        if mode == 'tr':
            with open(ROOT + 'tr_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                X_train_all = np.load(f)[:, :, :, args.NODC:]
            return X_train_all
        else:
            with open(ROOT + 'te_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                X_test_all = np.load(f)[:, :, :, args.NODC:]
            return X_test_all
    else:
        if mode == 'tr':
            with open(ROOT + 'tr_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                X_train_all = np.load(f)[:, :, :, args.NODC:][chosenidx]
            return X_train_all
        else:
            with open(ROOT + 'te_feature_Hop' + str(hopidx + 1) + '_AUG' + str(aug_idx) + '.npy', 'rb') as f:
                X_test_all = np.load(f)[:, :, :, args.NODC:][chosenidx]
            return X_test_all


def load_feat_allAUG_2(AUG_list, HOPLIST, tr_chosen=None, mode='tr',root_idx=0):
    tr_X_all = []

    for hop in HOPLIST:
        tr_X = []
        for aug in AUG_list:
            tr_X.append(get_feat_grouping_raw(hopidx=hop, mode=mode, ROOT=root[root_idx], aug_idx=aug, chosenidx=tr_chosen))
        tr_X = np.concatenate(tr_X, axis=0)

        ch = tr_X.shape[-1]
        X2_abs_all = np.copy(tr_X)
        if hop < 2:  #???
            if args.POOL == 2:
                X2_abs_all, absidx = [], []
                X2_ce_all, ce = [], []

                for c in tqdm(range(ch)):
                    X2_abs_tmp, absidx_tmp = LAYER.max_abs_pooling(data=tr_X[:,:,:,[c]], step=args.POOL)
                    X2_abs_all.append(X2_abs_tmp)

                X2_abs_all = np.array(X2_abs_all).squeeze()
                X2_abs_all = np.moveaxis(X2_abs_all,0,-1)

        tr_X_all.append(X2_abs_all)

    return tr_X_all


if __name__ == '__main__':
    # read data
    with open(args.dataroot + args.dataset + '_tr_label.npy', 'rb') as f:
        y_train = np.load(f)
    with open(args.dataroot + args.dataset + '_te_label.npy', 'rb') as f:
        y_test = np.load(f)

    y_train = y_train.squeeze().astype('int')
    y_test = y_test.squeeze().astype('int')

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Module 1 --------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    tr_X_LS, te_X_LS = [], []
    fs_list = {}

    # Load Spatial Saab feature
    for i in range(len(HOPLIST)):
        for k in range(len(root)):
            tr_X_LS.append(load_feat_allAUG_2(AUG_list, [HOPLIST[i]], mode='tr', tr_chosen=None, root_idx=k)[0])
            tr_NN, HH, WW, CC = tr_X_LS[-1].shape
            tr_X_LS[-1] = tr_X_LS[-1].reshape(tr_NN, HH, WW, -1)

        for k in range(len(root)):
            te_X_LS.append(load_feat_allAUG_2(AUG_list, [HOPLIST[i]], mode='te', tr_chosen=None, root_idx=k)[0])
            te_NN, HH, WW, CC = te_X_LS[-1].shape
            te_X_LS[-1] = te_X_LS[-1].reshape(te_NN, HH, WW, -1)

    # Extract Spectral Saab feature
    tr_X_GS, te_X_GS = [],[]
    for i in range(len(tr_X_LS)):
        with open(root[0] + 'gsModel_hop'+str(HOPLIST[i])+'.pkl', 'rb') as f:
            gs_model = pickle.load(f)
        tr_X_GS.append(GS.get_joint_gs_feat(tr_X_LS[i], mode='test', gs_model=gs_model))
        te_X_GS.append(GS.get_joint_gs_feat(te_X_LS[i], mode='test', gs_model=gs_model))

    # Flatten features
    for i in range(len(HOPLIST)):
        tr_X_LS[i] = tr_X_LS[i].reshape(tr_NN,-1)
        te_X_LS[i] = te_X_LS[i].reshape(te_NN,-1)
        tr_X_GS[i] = tr_X_GS[i].reshape(tr_NN,-1)
        te_X_GS[i] = te_X_GS[i].reshape(te_NN,-1)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Module 2 --------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    tr_X_all, te_X_all = [], []
    for i in range(len(tr_X_LS)):
        tr_tmp, te_tmp = [], []
        # joint spatial-spectral feature
        tr_X_i = np.concatenate((tr_X_LS[i], tr_X_GS[i]), axis=1)
        te_X_i = np.concatenate((te_X_LS[i], te_X_GS[i]), axis=1)

        # Feature selection
        if True: # train DFT loss and save
            _, dft_feat_loss, _ = FEAT.feature_selection(tr_X_i, y_train, FStype='DFT_entropy', thrs=1.0, B=args.B)
            with open(saveroot + 'dftloss_hop{}.npy'.format(i), 'wb') as f:
                np.save(f, dft_feat_loss)
        else: # directly load DFT loss
            with open(saveroot + 'dftloss_hop{}.npy'.format(i), 'rb') as f:
                dft_feat_loss = np.load(f)
        selected = np.argsort(dft_feat_loss)[:DIM_list[i]]

        tr_X_all.append(tr_X_i[:, selected])
        te_X_all.append(te_X_i[:, selected])


    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Module 3 --------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    tr_prob, te_prob = [],[]
    for hop in range(2):
        tr_X_ss = tr_X_all[hop].reshape(tr_NN, -1)
        te_X_ss = te_X_all[hop].reshape(te_NN, -1)

        # %%
        print(tr_X_ss.shape)
        print(te_X_ss.shape)

        clf = xgb.XGBClassifier(n_jobs=-1,
                             objective='multi:softprob',
                             tree_method='gpu_hist', #gpu_id=args.GPU,
                             max_depth=3, n_estimators=500,
                             # min_child_weight=2, gamma=0.01,
                             subsample=1.0, learning_rate=0.2,
                             nthread=12, colsample_bytree=1.0).fit(tr_X_ss,
                                                                  y_train.reshape(-1))#,
                                                                  # early_stopping_rounds=100,
                                                                  # eval_metric=['merror',
                                                                  #              'mlogloss'],
                                                                  # eval_set=[(tr_X_ss,y_train.reshape(-1)),
                                                                  #           (te_X_ss,y_test.reshape(-1))])

        tr_prob.append(clf.predict_proba(tr_X_ss))
        te_prob.append(clf.predict_proba(te_X_ss))

    # ensemble
    tr_prob, te_prob = [],[]
    for hop in range(2):
        tr_X_ss = tr_X_all[hop].reshape(tr_NN, -1)
        te_X_ss = te_X_all[hop].reshape(te_NN, -1)

    tr_prob = np.array(tr_prob)
    te_prob = np.array(te_prob)

    tr_prob = np.mean(tr_prob,axis=0)
    te_prob = np.mean(te_prob,axis=0)

    with open(saveroot + 'tr_prob_ensem.npy', 'wb') as f:
        np.save(f, np.array(tr_prob))

    with open(saveroot + 'te_prob_ensem.npy', 'wb') as f:
        np.save(f, np.array(te_prob))

    tr_acc = np.sum(np.argmax(tr_prob, axis=-1) == y_train.reshape(-1)) / y_train.size
    te_acc = np.sum(np.argmax(te_prob, axis=-1) == y_test.reshape(-1)) / y_test.size

    TIME2 = time.time()
    print("--- %s seconds ---" % (TIME2 - START_TIME))

    print('finish')





