import numpy as np 
import math
import cv2
# from dct import Smooth_Interpolation
from skimage.measure import block_reduce
from skimage.util import view_as_windows
from skimage import exposure
from scipy import ndimage
import lib.lib_confusing as CONF
import lib.lib_cross_entropy as CE
import lib.lib_stats as STAT

def min_ce_pooling(data, y, step=2, NUM_CLS=10):
    ch = data.shape[-1]
    data = view_as_windows(data, (1, step, step, ch), (1, step, step, ch)).squeeze()
    nn, hh, ww, _, _ = data.shape

    ce = STAT.Cross_Entropy(num_class=NUM_CLS, num_bin=2)
    feat_ce, _ = ce.get_all_CE(np.abs(data.reshape(nn,hh*ww*step*step)), y, mode='Bin2')

    feat_ce = feat_ce.reshape(hh,ww,step,step)
    feat_ce = np.repeat(feat_ce[np.newaxis, :, :, :, :], nn, axis=0)
    feat_ce = feat_ce.reshape(nn*hh*ww,step*step)
    data = data.reshape(nn*hh*ww,step*step)

    pool_idx = np.argmin(feat_ce,axis=-1)
    pooled = data[np.arange(pool_idx.size), pool_idx]

    pooled = pooled.reshape(nn, hh, ww, ch)
    pool_idx = pool_idx.reshape(nn, hh, ww)

    return pooled, pool_idx,feat_ce


def max_raw_pooling(data, step=2):
    ch = data.shape[-1]
    data = view_as_windows(data, (1, step, step, ch), (1, step, step, ch)).squeeze()
    nn, hh, ww, _, _ = data.shape
    data = data.reshape(nn * hh * ww, -1)
    pool_idx = np.argmax(data, axis=-1)
    pooled = data[np.arange(pool_idx.size), pool_idx]

    pooled = pooled.reshape(nn, hh, ww, ch)
    pool_idx = pool_idx.reshape(nn, hh, ww)

    return pooled, pool_idx

def max_abs_pooling(data, step=2):
    ch = data.shape[-1]
    data = view_as_windows(data, (1, step, step, ch), (1, step, step, ch)).squeeze()
    nn, hh, ww, _, _ = data.shape
    data = data.reshape(nn * hh * ww, -1)
    pool_idx = np.argmax(np.abs(data), axis=-1)
    pooled = data[np.arange(pool_idx.size), pool_idx]

    pooled = pooled.reshape(nn, hh, ww, ch)
    pool_idx = pool_idx.reshape(nn, hh, ww)

    return pooled, pool_idx


def minmax_normalize(data, single=False):
    '''
    single: normalize each single image or not
    '''
    if single == False:
        data_new = data - data.min()
        return data_new/(data_new.max()+1e-5)
    elif single==True:
        data_new = np.zeros((data.shape))
        for n in range(data.shape[0]):
            data_new[n] = data[n] - data[n].min()
            data_new[n] = data_new[n]/(data_new[n].max()+1e-5)
        return data_new

def augment_old(img, mode=1):
    aug_img = []
    
    if mode==1:# original
        aug_img = np.copy(img)
    elif mode==2:# mirror
        aug_img = mirror(img)
    elif mode==3:
        aug_img = shrink_resize(img,num=1)
    elif mode==4:
        tmp = shrink_resize(img,num=1)
        aug_img = mirror(tmp)
    elif mode==5:
        aug_img = shrink_resize(img,num=2)
    elif mode==6:
        tmp = shrink_resize(img,num=2)
        aug_img = mirror(tmp)
    elif mode==7:
        aug_img = shrink_resize(img,num=3)
    elif mode==8:
        tmp = shrink_resize(img,num=3)
        aug_img = mirror(tmp)
    elif mode==9:
        aug_img = np.concatenate((img,img[:,:,::-1,:]),axis=2)
        
    return aug_img


def augment(img, mode=1): ### March 03
    aug_img = np.copy(img)
    if mode>0:
        if mode==1:# mirror
            aug_img = mirror(aug_img)
        elif mode>1:
            if (mode%2)==1:
                aug_img = mirror(aug_img)
            if mode==2 or mode==3:
                #random crop
                aug_img = randomcrop_Square(aug_img)
            if mode==4 or mode==5:
                #random crop
                aug_img = randomcrop_Rect(aug_img)
            if mode==6 or mode==7:
                # contrast
                aug_img  = exposure.equalize_hist(aug_img)
                aug_img  = exposure.adjust_gamma(aug_img,gamma=0.5)
            if mode==8 or mode==9:
                #rotate clockwise
                aug_img = ndimage.rotate(aug_img.squeeze(), 10, axes=(1, 2), reshape=False)
            if mode==10 or mode==11:
                # rotate counter-clockwise
                aug_img = ndimage.rotate(aug_img.squeeze(), -10, axes=(1, 2), reshape=False)

            # elif mode==6 or mode==7:
            #     # random translation 
            #     aug_img = randomtranslate(aug_img)
            # elif mode==8 or mode==9:
                # standard color change using pca (alexnet)
    if len(aug_img.shape)<4:
        aug_img = aug_img[:,:,:,np.newaxis]
    return aug_img

# def augment(img, mode=1): ### April 12
#     aug_img = np.copy(img)
#     if mode>0:
#         if mode==1:# mirror
#             aug_img = mirror(aug_img)
#         elif mode>1:
#             if (mode%2)==1:
#                 aug_img = mirror(aug_img)
#             if mode==2 or mode==3:
#                 #random crop
#                 aug_img = randomcrop_Rect(aug_img)
#             if mode==4 or mode==5:
#                 # contrast
#                 aug_img  = exposure.equalize_hist(aug_img)
#                 aug_img  = exposure.adjust_gamma(aug_img,gamma=0.5)
#             if mode==6 or mode==7:
#                 # contrast
#                 for n in range(aug_img.shape[0]):
#                     aug_img[n] = coloraug(aug_img[n])
#             if mode==8 or mode==9:
#                 # contrast
#                 aug_img = randomcrop_Rect(aug_img)
#                 for n in range(aug_img.shape[0]):
#                     aug_img[n] = coloraug(aug_img[n])
#             # elif mode==6 or mode==7:
#             #     # random translation 
#             #     aug_img = randomtranslate(aug_img)
#             # elif mode==8 or mode==9:
#                 # standard color change using pca (alexnet)
#     return aug_img



def coloraug(original_image):
    renorm_image = np.reshape(original_image,(original_image.shape[0]*original_image.shape[1],3))
    renorm_image = renorm_image.astype('float32')
    mean = np.mean(renorm_image, axis=0)
    std = np.std(renorm_image, axis=0)
    
    #normalize
    renorm_image -= mean
    renorm_image /= std
   
    renorm_image_plot = renorm_image.reshape(original_image.shape[0], original_image.shape[1], -1)
    
    # svd
    cov = np.cov(renorm_image, rowvar=False) #covariance matrix
    lambdas, p = np.linalg.eig(cov) # eigenvector and eigenvalue
    alphas = np.random.normal(0.1,0.25, 3) #random weights
   
    delta = np.dot(p, alphas*lambdas) # eigenvector * (alpha * lambda)T

    pca_augmentation_version_renorm_image = renorm_image + delta
    #reconstruct
    pca_color_image = pca_augmentation_version_renorm_image * std + mean
    pca_color_image = np.maximum(np.minimum(pca_color_image, 1), 0)
    return pca_color_image.reshape(original_image.shape[0], original_image.shape[1], -1)


def augment_combine(img, mode=[1,2]):
    aug_img = []
    for m in mode:
        aug_img.append(augment(img,mode=m))
    return aug_img

def randomcrop_Square(img):
    newimg = []
    H_ori = img.shape[1]
    if img.shape[1]>16:
        H = [27,26]
    else:
        H = [15]
    for n in range(img.shape[0]):
        idx = np.random.permutation(len(H))[0]
        XRANGE = img.shape[1]-H[idx] +1
        x = np.random.permutation(XRANGE)[0]
        y = np.random.permutation(XRANGE)[0]
        cropped = img[n][x:(x+H[idx]), y:(y+H[idx])]
        newimg.append(cv2.resize(cropped, (H_ori,H_ori), interpolation=cv2.INTER_LANCZOS4))
    return np.array(newimg)

def randomcrop_Rect(img):
    newimg = []
    H_ori = img.shape[1]
    if img.shape[1]>16:
        H = [28,27,26]
        W = [28,27,26]
    else:
        H = [16,15]
        W = [16,15]
    for n in range(img.shape[0]):
        idx = np.random.permutation(len(H))[:2]
        XRANGE = img.shape[1]-H[idx[0]] +1
        YRANGE = img.shape[1]-W[idx[1]] +1
        x = np.random.permutation(XRANGE)[0]
        y = np.random.permutation(YRANGE)[0]
        cropped = img[n][x:(x+H[idx[0]]), y:(y+W[idx[1]])]
        newimg.append(cv2.resize(cropped, (H_ori,H_ori), interpolation=cv2.INTER_LANCZOS4))
    return np.array(newimg)

# def randomtranslate(img):
#     newimg = []
    
#     return newimg

def randomtranslate(img):
    newimg = []
    return newimg

def mirror(img):
    mirrored = img[:,:,::-1,:]
    return mirrored

def shrink_resize(img,num=1):
    n,hh,ww,c = img.shape
    shrink = img[:,num:(-1*num),num:(-1*num),:]
    new_img = myResize(shrink,hh,ww)
    return new_img


def myResize(x, H, W):
    if len(x.shape)>3:
        new_x = np.zeros((x.shape[0], H, W, x.shape[3]))
        for i in range(0, x.shape[0]):
            for j in range(0, x.shape[3]):
                new_x[i,:,:,j] = cv2.resize(x[i,:,:,j], (W,H), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_x = np.zeros((x.shape[0], H, W))
        for i in range(0, x.shape[0]):
            new_x[i] = cv2.resize(x[i], (W,H), interpolation=cv2.INTER_LANCZOS4)
    return new_x

def inv_blocking(blocks,winsize,stride,H,W):
    avgMask = np.zeros((H,W))
    for ii in range(blocks.shape[1]):
        for jj in range(blocks.shape[2]):
            avgMask[ii*stride:(ii*stride+winsize), jj*stride:(jj*stride+winsize)] += np.ones((winsize,winsize))
    avgMask = avgMask[np.newaxis,:,:]
    
    merged_mean = np.zeros((blocks.shape[0],H,W))
    for ii in range(blocks.shape[1]):
        for jj in range(blocks.shape[2]):
            merged_mean[:, ii*stride:(ii*stride+winsize), jj*stride:(jj*stride+winsize)] += blocks[:,ii,jj]
    merged_mean = np.divide(merged_mean, avgMask+1e-5)
    
#    merged_var= np.zeros((blocks.shape[0],H,W))
#    for ii in range(blocks.shape[1]):
#        for jj in range(blocks.shape[2]):
#            merged_var[:, ii*stride:(ii*stride+winsize), jj*stride:(jj*stride+winsize)] += np.power(blocks[:,ii,jj]-merged_mean[:,ii*stride:(ii*stride+winsize), jj*stride:(jj*stride+winsize)],2)
#    merged_var = np.divide(merged_var, avgMask+1e-5)
    
    return merged_mean#,merged_var    


def MaxPooling(x,step=2):
    if len(x.shape)<4:
        return block_reduce(x, (1, step, step), np.max)
    else:
        return block_reduce(x, (1, step, step, 1), np.max)

def MedianPooling(x,step=2):
    if len(x.shape)<4:
        return block_reduce(x, (1, step, step), np.median)
    else:
        return block_reduce(x, (1, step, step, 1), np.median)

def DCTpooling(x_all,step=2):
    if len(x_all.shape)<4:
        x_all = np.expand_dims(x_all,axis=-1)
    x_all = x_all.astype('float64')
#    X = cv2.resize(X,(320,320)).reshape(1,320,320,1).astype('float64')
    win = int(step*16)
    if step>1:
        stride = step
    else:
        stride = 1
    Yimg = np.zeros((x_all.shape[0],int(x_all.shape[1]//step), int(x_all.shape[2]//step)))
    for i in range(x_all.shape[0]):   
        X = view_as_windows(x_all[[i]], (1,win,win,1), (1,stride,stride,1)).squeeze()
        X = X[np.newaxis,:,:,:,:]
        n,h,w,_,_ = X.shape
        X = X.reshape(-1,win,win,x_all.shape[-1])
        si = Smooth_Interpolation(initN=win, targetN=int(win//step), mode='block')
    #    si = Smooth_Interpolation(initN=8, targetN=4, mode='block')
        Y = si.transform(X).squeeze()
        Y = Y.reshape(n,h,w,int(win//step),int(win//step))
        Yimg[i] = inv_blocking(Y,int(win//step),int(stride//step),int(x_all.shape[1]//step), int(x_all.shape[2]//step))
    
    if len(Yimg.shape)<4:
        Yimg = Yimg[:,:,:,np.newaxis]
    return Yimg

def DirectPooling(x, step=2):
    if len(x.shape)<4:
        return x[:,::step,::step]
    else:
        return x[:,::step,::step,:]

def AvgPooling(x,step=2):
    if len(x.shape)<4:
        return block_reduce(x, (1, step, step), np.mean)
    else:
        return block_reduce(x, (1, step, step, 1), np.mean)

def Project_concat(feature):
    dim = 0
    for i in range(len(feature)):
        dim += feature[i].shape[3]
        feature[i] = np.moveaxis(feature[i],0,2)
    result = np.zeros((feature[0].shape[0],feature[0].shape[1],feature[0].shape[2],dim))
    for i in range(0,feature[0].shape[0]):
        for j in range(0,feature[0].shape[1]):
            scale = 1.
            for fea in feature:
                if scale == 1:
                    tmp = fea[i,j]
                else:
                    #print(i,j,i//scale,j//scale)
                    tmp = np.concatenate((tmp, fea[int(i//scale),int(j//scale)]), axis=1)
                scale *= 2
            result[i,j] = tmp
    result = np.moveaxis(result, 2, 0)
    return result