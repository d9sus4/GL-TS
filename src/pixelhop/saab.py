# v 2022.05.07
# Saab transformation, changed to sklearn pca
# modified from https://github.com/ChengyaoWang/PixelHop-_c-wSaab/blob/master/saab.py

import numpy as np
import numba
from sklearn.decomposition import PCA, IncrementalPCA


def pca_cal(X: np.ndarray, K: int): ## changed on May 07 to sklearn
    # cov = X.transpose() @ X
    # eva, eve = np.linalg.eigh(cov)
    # inds = eva.argsort()[::-1]
    # eva = eva[inds]
    # kernels = eve.transpose()[inds]
    # return kernels, eva / (X.shape[0] - 1), cov
    # pca = PCA(n_components=X.shape[-1], svd_solver='full').fit(X)
    # Use K for n_components instead of X.shape[-1] to compute only the first K components
    pca = IncrementalPCA(n_components=K, batch_size=100000)
    pca.fit(X)

    # Since you're now fitting only for K components, all returned components and variances are already the first K
    kernels = pca.components_  # This will have K rows
    eva = pca.explained_variance_  # This will have K elements

    # Compute the covariance matrix of the original data
    cov = np.cov(X, rowvar=False)

    return kernels, eva, cov

@numba.jit(forceobj = True, parallel = True)
def remove_mean(X: np.ndarray, feature_mean: np.ndarray):
    return X - feature_mean

@numba.jit(nopython = True, parallel = True)
def feat_transform(X: np.ndarray, kernel: np.ndarray):
    # print (X.shape, kernel.shape)
    return X @ kernel.transpose()


class Saab():
    def __init__(self, num_kernels=-1, needBias=True, bias=0, rmcp=False):
        self.rmcp = rmcp # remove constant patches from AC filter learning or not
        self.num_kernels = num_kernels 
        self.needBias = needBias
        self.Bias_previous = bias # bias calculated from previous
        self.Bias_current = [] # bias for the current Hop
        self.Kernels = []
        self.Mean0 = [] # feature mean of AC
        self.Energy = [] # kernel energy list
        self.cov = []
        self.trained = False

    def remove_constant_patch(self, X, thrs=1e-5):
        diff = np.sum(X, axis = 1)
        idx = np.argwhere(diff<thrs).squeeze()
        # X = np.delete(X, idx, axis=0)
        return X[diff>thrs]

    def fit(self, X): 
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')
        
        # add bias from the previous Hop
        if self.needBias == True:
            X += self.Bias_previous
            
        # remove DC, get AC components
        dc = np.mean(X, axis = 1, keepdims = True)
        X = remove_mean(X, dc)
        
        # calcualte bias at the current Hop
        self.Bias_current = np.max(np.linalg.norm(X, axis=1))
        
        # remove DC-only (constant) patches
        if self.rmcp == True:
            X = self.remove_constant_patch(X)
            
        # remove feature mean --> self.Mean0
        self.Mean0 = np.mean(X, axis = 0, keepdims = True)
        X = remove_mean(X, self.Mean0)

        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        
        # Rewritten PCA Using Numpy
        kernels, eva, cov = pca_cal(X, self.num_kernels)
        
        # Concatenate with DC kernel
        dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1]))# / np.sqrt(largest_ev)
        kernels = np.concatenate((dc_kernel, kernels[:-1]), axis = 0)
        
        # Concatenate with DC energy
        largest_ev = np.var(dc * np.sqrt(X.shape[-1]))  
        energy = np.concatenate((np.array([largest_ev]), eva[:-1]), axis = 0)
        energy = energy / np.sum(energy)
        
        # store
        self.Kernels, self.Energy, self.cov = kernels.astype('float32'), energy, cov
        self.trained = True


    def transform(self, X):
        # print(X.shape)
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')
        
        # add bias from the previous Hop
        if self.needBias == True:
            X += self.Bias_previous
            
        # remove feature mean of AC
        X = remove_mean(X, self.Mean0)
        
        # convolve with DC and AC filters
        X = feat_transform(X, self.Kernels)
        
        return X
    
    
if __name__ == "__main__":
    from sklearn import datasets
    import warnings
    warnings.filterwarnings("ignore")
    
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(data.shape))
        
    X = data.copy()
    X = X.reshape(X.shape[0], -1)[0:100]
    
    saab = Saab(num_kernels=-1, needBias=True, bias=0)
    saab.fit(X)
    
    Xt = saab.transform(X)
