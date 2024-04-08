# Yijing 2022.2.9
# Get joint-spatial-spectral Global Saab feature

import numpy as np
from sklearn.decomposition import PCA
from lib.simple_saab_sklearn import SimpSaab
from tqdm import tqdm

def get_joint_gs_feat(feat,abs=1,mode='train',gs_model=None): ### This is the main function: get joint spatial-spectral feature
    if mode=='train':
        if abs == 1:
            feat = np.abs(feat)
        _,gs_model, gs_feat = cwGlobalSaab(feat,mode='train')
        if abs == 1:
            gs_model['abs']=True
        else:
            gs_model['abs']=False

        return gs_model, gs_feat
    else:
        if gs_model['abs'] == True:
            feat = np.abs(feat)
        _,gs_feat = cwGlobalSaab(feat,mode='test',gs_model=gs_model)
        return gs_feat

def get_joint_disc_gs_feat(feat,ce_map=None,abs=1,mode='train',sp_mode=1,thrs=0.5,gs_model=None): ### This is the main function: get joint spatial-spectral feature
    if mode=='train':
        if abs == 1:
            feat = np.abs(feat)
        # _,gs_model, gs_feat = cwDiscGlobalSaab(feat,ce_map=ce_map,mode='train',sp_mode=sp_mode,thrs=thrs)
        _,gs_model, gs_feat = cwDiscGlobalSaab_selective(feat,ce_map=ce_map,mode='train',sp_mode=sp_mode,thrs=thrs)
        if abs == 1:
            gs_model['abs']=True
        else:
            gs_model['abs']=False
        return gs_model, gs_feat
    else:
        if gs_model['abs'] == True:
            feat = np.abs(feat)
        # _,gs_feat = cwDiscGlobalSaab(feat,mode='test',gs_model=gs_model)
        _,gs_feat = cwDiscGlobalSaab_selective(feat,mode='test',gs_model=gs_model)
        return gs_feat

def select_sp(ce_map,sp_mode=1, thrs=0.5):
    ce_map = ce_map.reshape(-1)
    if sp_mode==1: # based on percentage
        selected = np.sort(np.argsort(ce_map)[:int(thrs*ce_map.size)])
    elif sp_mode==2: # based on ce threshold
        selected = np.argwhere(ce_map<thrs).reshape(-1)
    return selected

def cwDiscGlobalSaab(feat, ce_map=None, mode='train',sp_mode=1,thrs=[0.5], gs_model=None):
    if mode == 'train':
        ## TRAIN
        tr_N, H, W, Ch = feat.shape
        print('PCA')
        gs_model = {}
        samples_reduced = []
        tr_coef_var_per_ch = []
        for ch_idx in tqdm(range(Ch)):
            samples_ch = feat[:, :, :, [ch_idx]]
            samples_ch = samples_ch.reshape(-1, H * W)

            gs_model['selected_ch' + str(ch_idx)] = select_sp(ce_map[:,:,ch_idx],sp_mode=sp_mode,thrs=thrs[ch_idx])
            samples_ch = samples_ch[:,gs_model['selected_ch' + str(ch_idx)]]
            print(gs_model['selected_ch' + str(ch_idx)].size)

            gs_model['ch' + str(ch_idx)] = SimpSaab()
            gs_model['ch' + str(ch_idx)].fit(samples_ch)
            # # # # # # #
            tmp = gs_model['ch' + str(ch_idx)].transform(samples_ch)
            samples_reduced.append(tmp)
            coef_var = np.var(tmp, axis=0)
            tr_coef_var_per_ch.append(coef_var)

        samples_reduced = np.concatenate(samples_reduced,axis=-1)
        # samples_reduced = np.moveaxis(samples_reduced,0,-1)
        print('Global-saab finished')

        return tr_coef_var_per_ch, gs_model, samples_reduced

    else:
        ## TEST
        te_N, H, W, Ch = feat.shape
        print('PCA')
        samples_reduced = []
        te_coef_var_per_ch = []
        for ch_idx in range(Ch):
            samples_ch = feat[:, :, :, [ch_idx]]
            samples_ch = samples_ch.reshape(-1, H * W)
            samples_ch = samples_ch[:,gs_model['selected_ch' + str(ch_idx)]]

            tmp = gs_model['ch' + str(ch_idx)].transform(samples_ch)
            samples_reduced.append(tmp.reshape(te_N, -1))
            coef_var = np.var(tmp, axis=0)
            te_coef_var_per_ch.append(coef_var)

        samples_reduced = np.concatenate(samples_reduced,axis=-1)
        # samples_reduced = np.moveaxis(samples_reduced,0,-1)

        print('Globa-Saab finished')

        return te_coef_var_per_ch, samples_reduced


def cwDiscGlobalSaab_selective(feat, ce_map=None, mode='train',sp_mode=1,thrs=[0.5], gs_model=None): # global thrs, only some channels pass to global saab
    if mode == 'train':
        ## TRAIN
        tr_N, H, W, Ch = feat.shape
        print('PCA')
        gs_model = {}
        samples_reduced = []
        tr_coef_var_per_ch = []
        for ch_idx in tqdm(range(Ch)):
            if np.sum(ce_map[:,:,ch_idx]<thrs[ch_idx])>5:
                samples_ch = feat[:, :, :, [ch_idx]]
                samples_ch = samples_ch.reshape(-1, H * W)

                gs_model['selected_ch' + str(ch_idx)] = select_sp(ce_map[:,:,ch_idx],sp_mode=sp_mode,thrs=thrs[ch_idx])
                samples_ch = samples_ch[:,gs_model['selected_ch' + str(ch_idx)]]
                print(gs_model['selected_ch' + str(ch_idx)].size)

                gs_model['ch' + str(ch_idx)] = SimpSaab()
                gs_model['ch' + str(ch_idx)].fit(samples_ch)
                # # # # # # #
                tmp = gs_model['ch' + str(ch_idx)].transform(samples_ch)
                samples_reduced.append(tmp)
                coef_var = np.var(tmp, axis=0)
                tr_coef_var_per_ch.append(coef_var)
            else:
                gs_model['selected_ch' + str(ch_idx)] = None

        samples_reduced = np.concatenate(samples_reduced,axis=-1)
        # samples_reduced = np.moveaxis(samples_reduced,0,-1)
        print('Global-saab finished')

        return tr_coef_var_per_ch, gs_model, samples_reduced

    else:
        ## TEST
        te_N, H, W, Ch = feat.shape
        print('PCA')
        samples_reduced = []
        te_coef_var_per_ch = []
        for ch_idx in range(Ch):
            if gs_model['selected_ch' + str(ch_idx)] is not None:
                print(ch_idx)
                samples_ch = feat[:, :, :, [ch_idx]]
                samples_ch = samples_ch.reshape(-1, H * W)
                samples_ch = samples_ch[:,gs_model['selected_ch' + str(ch_idx)]]

                tmp = gs_model['ch' + str(ch_idx)].transform(samples_ch)
                samples_reduced.append(tmp.reshape(te_N, -1))
                coef_var = np.var(tmp, axis=0)
                te_coef_var_per_ch.append(coef_var)

        samples_reduced = np.concatenate(samples_reduced,axis=-1)
        # samples_reduced = np.moveaxis(samples_reduced,0,-1)

        print('Globa-Saab finished')

        return te_coef_var_per_ch, samples_reduced


def cwGlobalSaab(feat,mode='train',gs_model=None):
    assert (len(feat.shape) == 4), "input feature dimension incorrect! Expect to have 4 dimensions."
    if mode == 'train':
        ## TRAIN
        tr_N, H, W, Ch = feat.shape
        print('PCA')
        gs_model = {}
        samples_reduced = []
        tr_coef_var_per_ch = []
        for ch_idx in tqdm(range(Ch)):
            samples_ch = feat[:, :, :, [ch_idx]]
            samples_ch = samples_ch.reshape(-1, H * W)
            gs_model['ch' + str(ch_idx)] = SimpSaab()
            gs_model['ch' + str(ch_idx)].fit(samples_ch)
            # # # # # # #
            tmp = gs_model['ch' + str(ch_idx)].transform(samples_ch)
            samples_reduced.append(tmp)
            coef_var = np.var(tmp, axis=0)
            tr_coef_var_per_ch.append(coef_var)

        samples_reduced = np.array(samples_reduced)
        samples_reduced = np.moveaxis(samples_reduced,0,-1)
        print('Global-saab finished')

        return tr_coef_var_per_ch, gs_model, samples_reduced

    else:
        ## TEST
        te_N, H, W, Ch = feat.shape
        print('PCA')
        samples_reduced = []
        te_coef_var_per_ch = []
        for ch_idx in range(Ch):
            samples_ch = feat[:, :, :, [ch_idx]]
            samples_ch = samples_ch.reshape(-1, H * W)
            tmp = gs_model['ch' + str(ch_idx)].transform(samples_ch)
            samples_reduced.append(tmp.reshape(te_N, -1))
            coef_var = np.var(tmp, axis=0)
            te_coef_var_per_ch.append(coef_var)

        samples_reduced = np.array(samples_reduced)
        samples_reduced = np.moveaxis(samples_reduced,0,-1)

        print('Globa-Saab finished')

        return te_coef_var_per_ch, samples_reduced

