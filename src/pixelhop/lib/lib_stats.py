# v2021.01.22 library for data analysis
'''
CE_v1, CE_v2, ANOVA, multiANOVA2, W-dist, multi W-dist
'''

import numpy as np
from sklearn.metrics import log_loss as LL
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans,MiniBatchKMeans
import matplotlib.pyplot as plt
import math
import sklearn
import numba
import multiprocessing as mp
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from lib.lib_cross_entropy import cross_entropy,cal_weighted_CE
from tqdm import tqdm

from scipy.stats import f_oneway
# from itertools import product
import itertools
import threading
import logging



def classwise_distribution_curve(X_projected, y, K=32, bound=None):
    bins = np.arange(X_projected.min(), X_projected.max() + 0.01, (X_projected.max() - X_projected.min()) / K)
    # bins = model[index[i]][0]
    classwise = []
    for c in range(10):
        tmp, _, _ = plt.hist(X_projected[y == c], bins=bins)
        classwise.append(tmp)
        plt.close()
    ####################2
    plt.figure()
    for c in range(10):
        plt.plot(classwise[c], label=str(c))
    if bound is not None:
        diff = np.abs(bins-bound)
        idx = np.argmin(diff)
        if bound>bins[idx]:
            plt.axvline(x=(bound-bins[idx])/(bins[idx+1]-bins[idx])+idx,linestyle='--')
        else:
            plt.axvline(x=(bound-bins[idx-1])/(bins[idx]-bins[idx-1])+idx-1,linestyle='--')

    plt.xticks(np.arange(bins.size)[::5], np.round(bins[::5], decimals=2))
    plt.legend()
    plt.suptitle('Class-wise distribution')

    plt.show()
    return classwise


class Cross_Entropy():
    def __init__(self, num_class, num_bin=10):
        self.num_class = (int)(num_class)
        self.num_bin = (int)(num_bin)

    def bin_process(self, x ,y):
        if np.max(x) ==  np.min(x):
            return -1*np.ones(self.num_bin)
        x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (self.num_bin)
        mybin = np.zeros((self.num_bin, self.num_class))
        b = x.astype('int64')
        b[b == self.num_bin] -= 1
        for i in range(b.shape[0]):
            mybin[b[i], y[i]] += 1.
        for l in range(0, self.num_class):
            p = np.array(y[ y==l ]).shape[0]
            mybin[:, l] /= (float)(p)
        return np.argmax(mybin, axis=1)

    def bin_process2(self, x ,y):
        if np.max(x) ==  np.min(x):
            return x.astype('int64')
        x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (self.num_bin)
        mybin = np.zeros((self.num_bin, self.num_class))
        b = x.astype('int64')
        b[b == self.num_bin] -= 1
        # for i in range(b.shape[0]):
        #     mybin[b[i], y[i]] += 1.
        return b

    def lloyd_max(self,x_points,num_bin = 10):
        interval = (x_points.max() - x_points.min())/num_bin
        init = [x_points.min()+interval/2]
        for i in range(num_bin-1):
            init.append(init[i] + interval)
        init = np.array(init).reshape(-1,1)
        kmean = KMeans(n_clusters=num_bin,init=init,n_jobs=-1).fit(x_points.reshape(-1,1))
        centroids = kmean.cluster_centers_
        centroids = centroids.squeeze()

        centroids = np.sort(centroids)

        boundary = []
        boundary.append(-1*np.float('inf'))
        for i in range(num_bin-1):
            boundary.append((centroids[i]+centroids[i+1])/2)
        boundary.append(np.float('inf'))

        if False:
            plt.hist(x_points.squeeze(), bins=boundary)
            for i in range(num_bin+1):
                plt.axvline(x=boundary[i].squeeze(), color='orange')#, linestyle='--')
            plt.show()

        return np.sort(np.array(boundary)), kmean

    def bin_process3(self,x,y):
        if np.max(x) ==  np.min(x):
            return np.zeros(x.shape[0]).astype('int64') #x.astype('int64')
        # candidates, _ = self.lloyd_max(x, num_bin=8)
        # kmean = MiniBatchKMeans(n_clusters=8, batch_size=1000).fit(x.reshape(-1, 1))
        # candidates = kmean.cluster_centers_
        # candidates = np.sort(candidates.squeeze())
        # candidates = candidates[1:-1]
        K = 8
        candidates = np.arange(np.min(x),np.max(x),(np.max(x)-np.min(x))/K)
        candidates = candidates[1:-1]

        wCE_i = np.zeros(candidates.shape[0])
        for k in range(candidates.shape[0]):
            wCE_i[k] = cal_weighted_CE(x, y, candidates[k],num_cls=self.num_class)
        best_bound = candidates[np.argmin(wCE_i)]

        # if True:
        #     classwise_distribution_curve(x,y,bound=best_bound)

        bin_labels = np.zeros(x.shape[0]).astype('int64')
        bin_labels[x<best_bound] = 0
        bin_labels[x>=best_bound] = 1
        return bin_labels

    def kmeans_process(self, x, y):
        kmeans = KMeans(n_clusters=self.num_bin, random_state=0).fit(x.reshape(1,-1))
        mybin = np.zeros((self.num_bin, self.num_class))
        b = kmeans.labels_.astype('int64')
        b[b == self.num_bin] -= 1
        for i in range(b.shape[0]):
            mybin[b[i], y[i]] += 1.
        for l in range(0, self.num_class):
            p = np.array(y[ y==l ]).shape[0]
            mybin[:, l] /= (float)(p)
        return np.argmax(mybin, axis=1)

    def assign_bins(self, x, y):
        # mybin = self.bin_process2(x[:,0], y)
        mybin = self.bin_process3(x[:,0], y)
        return mybin

    def compute_prob(self, x, y):
        prob = np.zeros((self.num_class, x.shape[1]))
        for k in range(0, x.shape[1]):
            mybin = self.bin_process(x[:,k], y[:,0])
            #mybin = self.kmeans_process(x[:,k], y[:,0])
            for l in range(0, self.num_class):
                p = mybin[mybin == l]
                p = np.array(p).shape[0]
                prob[l, k] = p / (float)(self.num_bin)
        return prob

    def cal_entropy(self,prob, num_cls=10):
        prob_tmp = np.copy(prob)
        prob_tmp[prob_tmp == 0] = 1
        tmp = np.sum(-1 * prob * np.log(prob_tmp), axis=-1)
        return tmp / np.log(num_cls)

    def Bin_Cross_Entropy(self, x, y, class_weight=None):
        x = x.astype('float64')
        y = y.astype('int64')
        y = y.reshape(-1, 1)
        prob = self.compute_prob(x, y)
        prob = -1 * np.log10(prob + 1e-5) / np.log10(self.num_class)
        y = np.moveaxis(y, 0, 1)
        H = np.zeros((self.num_class, x.shape[1]))
        for c in range(0, self.num_class):
            yy = y == c
            p = prob[c].reshape(prob.shape[1], 1)
            p = p.repeat(yy.shape[1], axis=1)
            H[c] += np.mean(yy * p, axis=1)
        if class_weight is not None:
            class_weight = np.array(class_weight)
            H *= class_weight.reshape(class_weight.shape[0],1) * self.num_class
        return np.mean(H, axis=0)

    def Bin_Cross_Entropy2(self, x, y, class_weight=None):
        x = x.astype('float64')
        y = y.astype('int64')
        y = y.squeeze()
        # prob = self.compute_prob(x, y)
        klabels = self.assign_bins(x,y)
        if np.unique(klabels).size>1:
            prob = np.zeros((self.num_bin, self.num_class))
            for i in range(self.num_bin):
                idx = (klabels == i)
                if idx.size>0:
                    tmp = y[idx]
                    for j in range(self.num_class):
                        # prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y==j].shape[0]) + 1e-5)
                        prob[i, j] = tmp.tolist().count(j)/ (y.tolist().count(j) + 1e-5)

            prob = (prob)/(np.sum(prob, axis=1,keepdims=True) + 1e-5)
            y = np.eye(self.num_class)[y.reshape(-1)]
            probab = prob[klabels]
            # entropy = self.cal_entropy(prob,num_cls=np.unique(y).size)
            dominant_class = np.argmax(prob, axis=-1)
        else:
            probab = (1.0/self.num_class)*np.ones((x.shape[0],self.num_class))
            dominant_class = 0
            y = np.eye(self.num_class)[y.reshape(-1)]

        return cross_entropy(probab,y), dominant_class

    # new cross entropy
    def KMeans_Entropy(self, X, Y):
        if np.unique(Y).shape[0] == 1: #alread pure
            return 0
        if X.shape[0] < self.num_bin:
            return -1

        interval = (X.max() - X.min()) / self.num_bin
        init = [X.min() + interval / 2]
        for i in range(self.num_bin - 1):
            init.append(init[i] + interval)
        init = np.array(init).reshape(-1, 1)

        kmeans = MiniBatchKMeans(n_clusters=self.num_bin, init=init, batch_size=10000).fit(X)

        prob = np.zeros((self.num_bin, self.num_class))
        num = []
        for i in range(self.num_bin):
            idx = (kmeans.labels_ == i)
            num.append(np.sum((kmeans.labels_ == i)))
            tmp = Y[idx]
            for j in range(self.num_class):
                # prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y==j].shape[0]) + 1e-5)
                prob[i, j] = tmp.tolist().count(j) / (Y.tolist().count(j) + 1e-5)
        num = np.array(num).squeeze()
        num = num/np.sum(num)
        num = num.reshape(1,-1)
        prob = (prob)/(np.sum(prob, axis=1,keepdims=True) + 1e-5)
        # y = np.eye(self.num_class)[Y.reshape(-1)]
        # probab = prob[kmeans.labels_]
        entropy = self.cal_entropy(prob,num_cls=10)
        weighted_entropy = num @ (entropy.reshape(-1,1))
        return weighted_entropy


    def KMeans_Cross_Entropy(self, X, Y):
        if np.unique(Y).shape[0] == 1: #alread pure
            return 0
        if X.shape[0] < self.num_bin:
            return -1

        interval = (X.max() - X.min()) / self.num_bin
        init = [X.min() + interval / 2]
        for i in range(self.num_bin - 1):
            init.append(init[i] + interval)
        init = np.array(init).reshape(-1, 1)

        kmeans = MiniBatchKMeans(n_clusters=self.num_bin, init=init, batch_size=1000).fit(X)

        prob = np.zeros((self.num_bin, self.num_class))
        for i in range(self.num_bin):
            idx = (kmeans.labels_ == i)
            tmp = Y[idx]
            for j in range(self.num_class):
                # prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y==j].shape[0]) + 1e-5)
                prob[i, j] = tmp.tolist().count(j) / (Y.tolist().count(j) + 1e-5)

        prob = (prob)/(np.sum(prob, axis=1,keepdims=True) + 1e-5)
        y = np.eye(self.num_class)[Y.reshape(-1)]
        probab = prob[kmeans.labels_]

        return cross_entropy(probab,y) #sklearn.metrics.log_loss(y, probab)/math.log(self.num_class)


    def get_all_Entropy(self, X, Y, mode='Kmeans'):
        '''
        Parameters
        ----------
        X : TYPE
            shape (N, M).
        Y : TYPE
            shape (N).
        mode : TYPE, optional
            'Bin' or 'Kmeans'. The default is 'Kmeans'.

        Returns
        -------
        feat_ce: CE for all the feature dimensions. The smaller the better

        '''
        feat_ce = np.zeros(X.shape[-1])
        for k in range(X.shape[-1]):
            if mode=='Bin':
                feat_ce[k] = self.Bin_Entropy(X[:,[k]], Y)
            else:
                feat_ce[k] = self.KMeans_Entropy(X[:,[k]], Y)
        return feat_ce

    def get_all_CE(self, X, Y, mode='Kmeans'):
        '''
        Parameters
        ----------
        X : TYPE
            shape (N, M).
        Y : TYPE
            shape (N).
        mode : TYPE, optional
            'Bin' or 'Kmeans'. The default is 'Kmeans'.

        Returns 
        -------
        feat_ce: CE for all the feature dimensions. The smaller the better

        '''
        feat_ce = np.zeros(X.shape[-1])
        dominant = []
        for k in tqdm(range(X.shape[-1])):
            # if k==532:
            #     print('a')
            if mode=='Bin':
                feat_ce[k] = self.Bin_Cross_Entropy(X[:,[k]], Y)
            elif mode=='Bin2':
                feat_ce[k], dominant_k = self.Bin_Cross_Entropy2(X[:,[k]], Y)
                dominant.append(dominant_k)
            else:
                feat_ce[k] = self.KMeans_Cross_Entropy(X[:,[k]], Y)
        return feat_ce,np.array(dominant)

def ANOVA(X,y):
    y_uniq = np.unique(y).squeeze()
    F_value = np.zeros(X.shape[1])
    X_0 = X[y==y_uniq[0]]
    X_1 = X[y==y_uniq[1]]
    
    for c in range(X.shape[1]):
        mu = np.mean(X[:,c])
        mu_0 = np.mean(X_0[:,c])
        mu_1 = np.mean(X_1[:,c])
            
        SSB = X_0.shape[0]*np.power(mu_0-mu,2) + X_1.shape[0]*np.power(mu_1-mu,2)
        MSB = SSB/(2-1)
        SSE = np.sum(np.power(X_0[:,c]-mu_0,2)) + np.sum(np.power(X_1[:,c]-mu_1,2))
        MSE = SSE/(X.shape[0])#-2)
        
        F_value[c] = MSB/MSE 
    return F_value

# def ANOVA_multiclass(X,y):
#     y_uniq = np.unique(y).squeeze()
#     F_all = []
#     k=0
#     for c1 in range(y_uniq.size):
#         for c2 in range(y_uniq.size):
#             if c1>c2:
#                 k+=1
#                 F_value = np.zeros(X.shape[1])
#                 X_0 = X[y==y_uniq[c1]]
#                 X_1 = X[y==y_uniq[c2]]
                
#                 for ch in range(X.shape[1]):
#                     mu = np.mean(X[:,ch])
#                     mu_0 = np.mean(X_0[:,ch])
#                     mu_1 = np.mean(X_1[:,ch])
                        
#                     SSB = X_0.shape[0]*np.power(mu_0-mu,2) + X_1.shape[0]*np.power(mu_1-mu,2)
#                     MSB = SSB/(2-1)
#                     SSE = np.sum(np.power(X_0[:,ch]-mu_0,2)) + np.sum(np.power(X_1[:,ch]-mu_1,2))
#                     MSE = SSE/(X.shape[0])#-2)
                    
#                     F_value[ch] = MSB/MSE 
                    
#                 F_all.append(F_value)
                
#     F_all = np.array(F_all)
#     F_all = np.mean(F_all,axis=0)
#     return F_all

# @numba.jit(forceobj = True, parallel = True)
def ANOVA_multiclass2(X,y):
    y_uniq = np.unique(y).squeeze()
    NUM_CLS = y_uniq.size
    F_value = np.zeros(X.shape[1])

    for ch in range(X.shape[1]):
        mu = np.mean(X[:,ch])
        SSB = 0
        SSE = 0
        for c in y_uniq.tolist():
            X_c = X[y==c]
            mu_c = np.mean(X_c[:,ch])
            SSB += X_c.shape[0]*np.power(mu_c-mu,2)
            SSE += np.sum(np.power(X_c[:,ch]-mu_c,2))
        MSE = SSE/(X.shape[0])#-2)
        MSB = SSB/(NUM_CLS-1)
        F_value[ch] = MSB/MSE 
    return F_value

def ANOVA_multiclass3(X,y, bestK=10):
    fs = SelectKBest(score_func=f_classif, k=bestK)
    fs.fit(X, y)
    X = fs.transform(X)
    return X, fs

def ANOVA_multiclass_singlechannel(X,y):
    y_uniq = np.unique(y).squeeze()
    NUM_CLS = y_uniq.size
    # F_value = np.zeros(X.shape[1])
    mu = np.mean(X)
    SSB = 0
    SSE = 0
    for c in y_uniq.tolist():
        X_c = X[y==c]
        mu_c = np.mean(X_c)
        SSB += X_c.shape[0]*np.power(mu_c-mu,2)
        SSE += np.sum(np.power(X_c-mu_c,2))
    MSE = SSE/(X.shape[0])#-2)
    MSB = SSB/(NUM_CLS-1)
    F_value = MSB/MSE 
    return F_value

def multi_run_wrapper(args):
   return ANOVA_multiclass_singlechannel(*args)

def ANOVA_multiclass2_par(X, y, N=8):
    with mp.Pool(processes = N) as p:
        # results = p.map(ANOVA_multiclass_singlechannel, [X[:,ch] for ch in range(X.shape[1])], [y for ch in range(X.shape[1])])   
        results = p.map(multi_run_wrapper, [(X[:,ch],y) for ch in range(X.shape[1])])   
        
    return np.array(results).squeeze()
    

# def ANOVA_multiclass2_multithread(X, y, N=8):
#     threads = list()
#     BS = int(np.ceil(X.shape[1]/np.float(N)))
#     for nn in range(N):
#         tmp = threading.Thread(target=ANOVA_multiclass2, args=(X[:,nn*BS:(nn+1)*(BS)],y))
#         threads.append(tmp)
#         tmp.start()
#     for index, thread in enumerate(threads):
#         logging.info("Main    : before joining thread %d.", index)
#         thread.join()
#         logging.info("Main    : thread %d done", index)
#     return threads


def clf_CE(y, pred_prob, num_cls=2):
    y = y.astype('int64')
    return LL(np.eye(num_cls)[y], pred_prob)/np.log(num_cls)

def W_dist(d1, d2):
    return wasserstein_distance(d1, d2)
    
def W_dist_multiclass(d_all, y_all):
    y_all_unique = np.unique(y_all).tolist()
    dist = 0
    k=0
    for c1 in y_all_unique:
        for c2 in y_all_unique:
            if c1>c2:
                k+=1
                dist += wasserstein_distance(d_all[y_all==c1], d_all[y_all==c2])
    mDist = dist/k
    return mDist

# if __name__ == '__main__':
