"""
# v 2022.02.03
# new feature extraction
# save feature for each augmentation separately
"""
import numpy as np
from skimage.util import view_as_windows
from pixelhop import Pixelhop
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings, gc
import time
# from memory_profiler import profile
import layer
import pickle
import os
import cv2
import lib.lib_confusing as CONF
import lib.lib_cross_entropy as CE
import argparse
import lib.semi_utils as SEMI
import lib.lib_global_saab as GS
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='DataLoader')
    parser.add_argument('--AUG', default='0', help='data root')
    parser.add_argument('--TH1', default=0.002, type=float,help='save npy file or not')
    parser.add_argument('--TH2', default=0.001, type=float, help='save png file or not')
    parser.add_argument('--train_M1',default=1, type=int, help='filename')
    parser.add_argument('--getoutput_M1',default=1, type=int, help='filename')
    parser.add_argument('--RES',default=32, type=int, help='filename')
    parser.add_argument('--layers',default=2, type=int, help='filename')
    parser.add_argument('--BS',default=60000, type=int, help='filename')
    parser.add_argument('--eachnum',default=16, type=int, help='filename')
    parser.add_argument('--pattern', default='0,0,0',help='saveroot')
    parser.add_argument('--WIN', default='5,5,5',help='saveroot')
    parser.add_argument('--dataset', default='mnist',help='saveroot')
    parser.add_argument('--dataroot', default='/project/jckuo_84/yijingya/dataset/',help='saveroot')
    parser.add_argument('--saveroot', default='/scratch1/yijingya/weak_sup/feat/0514_fullset_',help='saveroot')
    args = parser.parse_args()
    return args

args = parse_args()

BS = args.BS

pattern_list = [int(s) for s in args.pattern.split(',')]
AUG_list = [int(s) for s in args.AUG.split(',')]
win_list = [int(s) for s in args.WIN.split(',')]

saveroot = args.saveroot+'_{}/'.format(args.dataset)
if not os.path.isdir(saveroot):os.makedirs(saveroot)

def shuffle_data(X,y):
    shuffle_idx = np.random.permutation(y.size)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    return X, y

def select_balanced_subset(images, labels, use_num_images):
    '''
    select equal number of images from each classes
    '''
    num_total, H, W, C = images.shape
    num_class = np.unique(labels).size
    num_per_class = int(use_num_images/num_class)

    # Shuffle
    images, labels = shuffle_data(images, labels)
    
    selected_images = np.zeros((use_num_images, H, W, C))
    selected_labels = np.zeros(use_num_images)
    
    for i in range(num_class):
        selected_images[i*num_per_class:(i+1)*num_per_class] = images[labels==i][:num_per_class]
        selected_labels[i*num_per_class:(i+1)*num_per_class] = np.ones((num_per_class)) * i

    # Shuffle again
    selected_images, selected_labels = shuffle_data(selected_images, selected_labels)
    
    return selected_images, selected_labels


def Shrink(X, shrinkArg):
    pool = shrinkArg['pool']
    ch = X.shape[-1]
    # if pool>1:
        # if pool==2:
        #     X = block_reduce(X, (1, pool, pool, 1), np.max)
        # elif pool==3: # overlapping pool
        #     X = view_as_windows(X, (1,pool,pool,ch), (1,2,2,ch))
        #     X = X.reshape(X.shape[0],X.shape[1],X.shape[2],pool*pool,-1)
        #     X = np.max(X,axis=3)
    if pool==2:
        # X2_ce, ceidx = layer.min_ce_pooling(data=X, y=shrinkArg['label'], step=pool)

        # newly added in 1/11/24 -- 4 lines
        if X.shape[1]%2 != 0:
            X = np.concatenate([X, X[:, X.shape[1] // 2:X.shape[1] // 2 + 1, :]], axis=1)
        elif X.shape[2]%2 != 0:
            X = np.concatenate([X, X[:, :, X.shape[2] // 2:X.shape[2] // 2 + 1]], axis=2)

        X, _ = layer.max_abs_pooling(data=X, step=pool)
        X = np.abs(X)

    else:
        ceidx, absidx = None,None
    win = shrinkArg['win']
    stride = shrinkArg['stride']
    pad = shrinkArg['pad']
    # print(X.shape)
    if pad>0:
        X = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),shrinkArg['padmode']) #???? first Hop constant padding?

    X = view_as_windows(X, (1, win, win, ch), (1, stride, stride, ch))
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    return X#,ceidx, absidx

# example callback function for how to concate features from different hops
def Concat(X, concatArg):
    return X

def get_feat(X, num_layers=3, start=0, output_all=False):
    if output_all == True:
        output = []
        output.append(p2.transform_singleHop(X, layer=0))
    else:
        output = p2.transform_singleHop(X, layer=0)
    fwrite = open(saveroot + 'Hop1_dim' + str(output[-1].shape[-1]) + '.pkl', 'wb')
    pickle.dump([], fwrite)
    fwrite.close()

    if num_layers > 1:
        for i in range(num_layers-1):
            if output_all == True:
                tmp = p2.transform_singleHop(output[-1], layer=i+1)
                output.append(tmp)
                fwrite = open(saveroot + 'Hop' + str(i + 2) + '_dim' + str(tmp.shape[-1]) + '.pkl', 'wb')
                pickle.dump([], fwrite)
                fwrite.close()
                if ((start>0) and (i > 0)):
                    for kk in range(min(i, start)):
                        output[kk] = []
            else:
                output = p2.transform_singleHop(output, layer=i+1)

    return output

def save_feat(X, BS=10, mode='tr', saveroot='./', num_layers=4, start=0, output_all=True):
    N_Full = X.shape[0]
    if BS<(N_Full//len(AUG_list)):
        for i in range(N_Full//BS):
            tmp_output = get_feat(X[i*BS:(i+1)*BS], num_layers=num_layers, start=start, output_all=output_all)
            NUM_HOPS = len(tmp_output)
            for k in range(NUM_HOPS):
                if k > (start - 1):
                    with open(saveroot + mode + '_output_stage'+str(k+1)+'_'+str(i)+'.npy', 'wb') as f:
                        np.save(f, tmp_output[k])
                    fwrite = open(saveroot+'AUG'+str(len(AUG_list))+'_Hop'+str(k+1)+'_dim'+str(tmp_output[k].shape[-1])+'.pkl','wb')
                    pickle.dump([],fwrite)
                    fwrite.close()
            del tmp_output

        for k in range(num_layers):
            if k > (start-1):
                print(k)
                output = []
                for i in range(N_Full//BS):
                    with open(saveroot + mode + '_output_stage'+str(k+1)+'_'+str(i)+'.npy', 'rb') as f:
                        output.append(np.load(f))
                    with open(saveroot + mode + '_output_stage'+str(k+1)+'_'+str(i)+'.npy', 'wb') as f:
                        np.save(f, np.array([0]))
                output = np.concatenate(output,axis=0)#.reshape(sample_images.shape[0],-1)
                print(output.shape)

                if mode=='tr':
                    num_each_aug = tr_N
                else:
                    num_each_aug = te_N

                for aug in range(len(AUG_list)):
                    with open(saveroot + mode + '_feature_Hop'+str(k+1)+'_AUG'+str(AUG_list[aug])+'.npy', 'wb') as f:
                        np.save(f, output[aug*num_each_aug:(aug+1)*num_each_aug])

            # with open(saveroot + mode + '_feature_Hop'+str(k+1)+'_all_AUG'+str(len(AUG_list))+'.npy', 'wb') as f:
            #     np.save(f, output)
    else:
        output = get_feat(X, num_layers=num_layers, start=start, output_all=output_all)
        NUM_HOPS = len(output)
        if mode == 'tr':
            num_each_aug = tr_N
        else:
            num_each_aug = te_N

        for k in range(start,NUM_HOPS):
            for aug in range(len(AUG_list)):
                with open(saveroot + mode + '_feature_Hop' + str(k + 1) + '_AUG' + str(AUG_list[aug]) + '.npy', 'wb') as f:
                    np.save(f, output[k][aug * num_each_aug:(aug + 1) * num_each_aug])
                print(output[k][aug * num_each_aug:(aug + 1) * num_each_aug].shape)
                # fwrite = open(saveroot + 'AUG' + str(len(AUG_list)) + '_Hop' + str(k + 1) + '_dim' + str(output[k].shape[-1]) + '.pkl', 'wb')
                # pickle.dump([], fwrite)
                # fwrite.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    '''Load MNIST-10 data and split'''

    with open(args.dataroot + args.dataset + '_tr_img.npy', 'rb') as f:
        train_data_ori = np.load(f)
    with open(args.dataroot + args.dataset + '_te_img.npy', 'rb') as f:
        test_data_ori = np.load(f)
    with open(args.dataroot + args.dataset + '_tr_label.npy', 'rb') as f:
        train_labels = np.load(f)
    with open(args.dataroot + args.dataset + '_te_label.npy', 'rb') as f:
        test_labels = np.load(f)

            # dataset_mean = 0.1307
        # dataset_std = 0.3081
    # -----------Data Preprocessing-----------
    train_data_ori = train_data_ori[:,:,:,np.newaxis]
    test_data_ori = test_data_ori[:,:,:,np.newaxis]

    tr_N = train_data_ori.shape[0]
    te_N = test_data_ori.shape[0]

    print(train_labels[:10])
    print(test_labels[:10])
    train_data_ori = train_data_ori.astype('float32')#/255.
    test_data_ori = test_data_ori.astype('float32')#/255.

    if args.RES != 32:
        # if train_data_ori.shape[1] == 28:
        #     train_data_ori = np.pad(train_data_ori,((0,0),(2,2),(2,2),(0,0)))
        #     test_data_ori = np.pad(test_data_ori,((0,0),(2,2),(2,2),(0,0)))
        # plt.imshow(train_data_ori[927].squeeze(),cmap='gray'); plt.show()

        # reduce to 16x16
        train_data_ori_16 = np.zeros((tr_N,args.RES,args.RES))
        test_data_ori_16 = np.zeros((te_N,args.RES,args.RES))
        for n in range(tr_N):
            train_data_ori_16[n] = cv2.resize(train_data_ori[n],(args.RES,args.RES),interpolation=cv2.INTER_LANCZOS4)
        train_data_ori = np.copy(train_data_ori_16)
        train_data_ori[train_data_ori<0]=0

        for n in range(te_N):
            test_data_ori_16[n] = cv2.resize(test_data_ori[n],(args.RES,args.RES),interpolation=cv2.INTER_LANCZOS4)
        test_data_ori = np.copy(test_data_ori_16)
        test_data_ori[test_data_ori<0]=0

        # plt.imshow(train_data_ori_16[927].squeeze(),cmap='gray'); plt.show()

        # train_data_ori = (train_data_ori-dataset_mean)/dataset_std
        # test_data_ori = (test_data_ori-dataset_mean)/dataset_std

    if len(AUG_list)>1:    
        train_data_ori = layer.augment_combine(train_data_ori,mode=AUG_list)
        train_data_ori = np.concatenate(train_data_ori,axis=0)
        test_data_ori = layer.augment_combine(test_data_ori,mode=AUG_list)
        test_data_ori = np.concatenate(test_data_ori,axis=0)

    # if args.dataset == 'fashion':


    if True:
        train_data_ori[train_data_ori < 0] = 0
        test_data_ori[test_data_ori < 0] = 0

    train_data_ori = layer.minmax_normalize(train_data_ori,single=True)
    test_data_ori = layer.minmax_normalize(test_data_ori,single=True)

    if len(train_data_ori.shape)<4:
        train_data_ori = train_data_ori[:,:,:,np.newaxis]
        test_data_ori = test_data_ori[:,:,:,np.newaxis]

        
    train_data = np.copy(train_data_ori)
    test_data = np.copy(test_data_ori)
    #
    # if True:
    #     train_data[train_data < 1/255.] = 0
    #     test_data[test_data < 1/255.] = 0
        
    del train_data_ori, test_data_ori
    #%%
    '''Saab Unsupervised Feature Extraction - Module - 1'''

    # -----------set parameters-----------
    # read data
    print(" > This is a test example: ")
    SaabArgs = [{'num_AC_kernels':-1, 'needBias': False, 'cw': False},
                {'num_AC_kernels':-1, 'needBias': True,  'cw': True},
                {'num_AC_kernels': -1, 'needBias': True, 'cw': True}]

    shrinkArgs = [{'func':Shrink, 'win': win_list[0], 'label':train_labels,'stride': 1, 'pad':2, 'pool': 1, 'pattern': pattern_list[0],'padmode': 'constant'},
                {'func': Shrink, 'win': win_list[1], 'label':train_labels, 'stride': 1, 'pad':0, 'pool': 2, 'pattern': pattern_list[1],'padmode': 'reflect'},
                {'func': Shrink, 'win': win_list[2], 'label':train_labels, 'stride': 1, 'pad':0, 'pool': 2, 'pattern': pattern_list[2],'padmode': 'reflect'}]
    concatArg = {'func':Concat}

    #%% pixelhop train
    if args.train_M1:
        start = time.time()
    # -----------Module-(1)-----------
        p2 = Pixelhop(depth = args.layers,
                       TH1 = args.TH1,
                       TH2 = args.TH2,
                       SaabArgs = SaabArgs,
                       shrinkArgs = shrinkArgs,
                       concatArg = concatArg).fit(train_data[:tr_N])
        end = time.time()
        print(end - start)
        p2.save(saveroot + 'model')

    if args.getoutput_M1:
        if not os.path.isdir(saveroot+'/'): os.makedirs(saveroot+'/')

        p2 = Pixelhop(load=True).load(saveroot + 'model')
        #%% get feature
        start = time.time()
        save_feat(test_data, BS=BS, mode='te', saveroot=saveroot+'/', num_layers=args.layers, start=0, output_all=True)
        save_feat(train_data, BS=BS, mode='tr', saveroot=saveroot+'/', num_layers=args.layers, start=0, output_all=True)
        end = time.time()
        print(end - start)

        # ---- GS
        tr_X_ss_list,te_X_ss_list = [],[]
        for hop in range(2):
            with open(saveroot+'/' + 'tr_feature_Hop'+str(hop+1)+'_AUG0.npy', 'rb') as f:
                tr_tmp = np.load(f)
            with open(saveroot+'/' + 'te_feature_Hop'+str(hop+1)+'_AUG0.npy', 'rb') as f:
                te_tmp = np.load(f)
            if True:
                tr_abs_all,te_abs_all = [],[]
                for c in tqdm(range(tr_tmp.shape[-1])):
                    tr_abs_tmp, _ = layer.max_abs_pooling(data=tr_tmp[:, :, :, [c]], step=2)
                    tr_abs_all.append(tr_abs_tmp)
                    te_abs_tmp, _ = layer.max_abs_pooling(data=te_tmp[:, :, :, [c]], step=2)
                    te_abs_all.append(te_abs_tmp)

                tr_abs_all = np.array(tr_abs_all).squeeze()
                tr_abs_all = np.moveaxis(tr_abs_all, 0, -1)
                te_abs_all = np.array(te_abs_all).squeeze()
                te_abs_all = np.moveaxis(te_abs_all, 0, -1)

            tr_X_ss_list.append(tr_abs_all)
            te_X_ss_list.append(te_abs_all)
            print(tr_abs_all.shape,te_abs_all.shape)

        for hop in range(2):
            gs_model, _ = GS.get_joint_gs_feat(tr_X_ss_list[hop], mode='train',abs=1,gs_model=None)
            # tr_GS = GS.get_joint_gs_feat(tr_X_ss_list[hop], mode='test',gs_model=gs_model)
            # te_GS = GS.get_joint_gs_feat(te_X_ss_list[hop], mode='test',gs_model=gs_model)
            with open(saveroot+'/' + 'gsModel_hop'+str(hop)+'.pkl', 'wb') as f:
                pickle.dump(gs_model,f,protocol=2)
            #
            # with open(saveroot+'/' + 'tr_GS_hop'+str(hop)+'.npy', 'wb') as f:
            #     np.save(f,tr_GS)
            # with open(saveroot+'/'+ 'te_GS_hop'+str(hop)+'.npy', 'wb') as f:
            #     np.save(f,te_GS)
