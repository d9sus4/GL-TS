"""
Impute a preprocessed multivariate time sereis (missing values included) dataset by a specific strategy.
Keyword arguments are:
    - `fp`: input pickle file path (with 'data' and 'meta' fields).
    - `strat`: imputation strategy.

"""


import pandas as pd
import numpy as np
import pickle
import argparse
import os
import json
from glob import glob
from tqdm import tqdm
from dataclasses import dataclass, asdict
import numpy as np
from scipy import interpolate
from sklearn.cluster import KMeans
import xgboost as xgb

from .pixelhop.saab import Saab
from .utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Impute a preprocessed multivariate time series (missing values included) dataset.')
    parser.add_argument('--fp', type=str, default='./data/p19/p19_clean.pkl',
                        help='Path to the processed dataset hdf5 file.')
    parser.add_argument('--strat', choices=['zero', 'mean', 'forward', 'linear', 'spline', 'kmeans', 'xgboost'],
                        help='Choose imputation strategy.')
    parser.add_argument('--mrthold', type=float, default=1., help='Missing rate threshold for sensor dropping before imputing, default is 1 (no dropping)')
    parser.add_argument('--out_dir', type=str, default='./data/p19/',
                        help='Path to the directory where the output will be stored.')
    args = parser.parse_args()
    return args


@dataclass
class KMeansImputerConfig:
    W: int = 3
    stride: int = 1
    num_bins: int = 4
    num_kernels: int = 16
    num_clusters: int = 16

@dataclass
class XGBoostImputerConfig:
    W: int = 3 # 2W+1 window

def impute_all_zero(obs, mask=None):
    """Impute all NaNs with 0."""
    if mask is None:
        mask = ~np.isnan(obs)
    imp = np.copy(obs)
    imp[~mask] = 0
    return imp

def impute_mean(obs, mean, mask=None):
    """Impute all NaNs with the mean value of their respective sensor."""
    if mask is None:
        mask = ~np.isnan(obs)
    imp = np.copy(obs)
    for i in range(obs.shape[1]):
        imp[~mask[:, i], i] = mean[i]
    return imp

def impute_forward_fill(obs, mean):
    """Impute starting NaNs with mean, then forward fill the rest."""
    imp = np.copy(obs)
    for i in range(obs.shape[1]):
        if np.isnan(imp[0, i]):
            imp[0, i] = mean[i]
        for j in range(1, len(imp)):
            if np.isnan(imp[j, i]):
                imp[j, i] = imp[j-1, i]
    return imp

def impute_linear(obs, mean, mask=None, stamp=None):
    """Linear interpolation with time stamps and conditions for all-NaN columns or single non-NaN values."""
    if mask is None:
        mask = ~np.isnan(obs)
    imp = np.copy(obs)
    if stamp is None:
        stamp = np.arange(obs.shape[0])  # Use index as x-axis if stamp is None
    for i in range(obs.shape[1]):
        # Check the condition for all NaN column
        if not np.any(mask[:, i]):
            imp[:, i] = mean[i]
            continue  # Skip to next column

        non_nan_idx = np.where(mask[:, i])[0]
        
        # If only one non-NaN value in a column, fill the column with that value
        if len(non_nan_idx) == 1:
            imp[:, i] = imp[non_nan_idx, i]
            continue
        
        # Apply linear interpolation with extrapolation, considering time stamps
        f = interpolate.interp1d(stamp[non_nan_idx], imp[non_nan_idx, i], kind='linear', bounds_error=False, fill_value='extrapolate')
        imp[:, i] = f(stamp)
    return imp

def impute_spline(obs, mean, mask=None, stamp=None):
    """Cubic spline interpolation with time stamps for smoother imputation."""
    if mask is None:
        mask = ~np.isnan(obs)
    imp = np.copy(obs)
    if stamp is None:
        stamp = np.arange(obs.shape[0])  # Use index as x-axis if stamp is None
    for i in range(obs.shape[1]):
        # Check the condition for all NaN column
        if not np.any(mask[:, i]):
            imp[:, i] = mean[i]
            continue  # Skip to next column

        non_nan_idx = np.where(mask[:, i])[0]
        
        # If only one non-NaN value in a column, fill the column with that value
        if len(non_nan_idx) == 1:
            imp[:, i] = imp[non_nan_idx, i]
            continue

        # Apply cubic spline interpolation with extrapolation, considering time stamps
        try:
            f = interpolate.CubicSpline(stamp[non_nan_idx], imp[non_nan_idx, i], extrapolate=True)
            imp[:, i] = f(stamp)
        except ValueError:
            # Fallback to linear interpolation if CubicSpline cannot be used
            f = interpolate.interp1d(stamp[non_nan_idx], imp[non_nan_idx, i], kind='linear', bounds_error=False, fill_value='extrapolate')
            imp[:, i] = f(stamp)
            
    return imp


def calculate_imputation_confidence_global_window(mask, W):
    """
    Calculate the confidence of imputation for each position in the observation tensor
    considering a window across all sensors for each timestamp.

    Args:
    - mask (ndarray): The mask tensor indicating observed (True) vs imputed (False) values, shape (seq_len, num_sensors).
    - W (int): Half the window size used for calculating confidence.

    Returns:
    - conf (ndarray): The confidence tensor, shape (seq_len, num_sensors).
    """
    seq_len, num_sensors = mask.shape
    conf = np.zeros_like(mask, dtype=np.float32) # conf matrix, element is 1 where real data is observed
    conf_vec = np.zeros(seq_len, dtype=np.float32) # conf vector, shared by all tensors in a row
    # Iterate over each timestamp
    for time_idx in range(seq_len):
        # Determine window bounds for the timestamp
        start_idx = max(0, time_idx - W)
        end_idx = min(seq_len, time_idx + W + 1)  # +1 because the range is exclusive at the end

        # Calculate the confidence level based on the mask values within the window across all sensors
        window_mask = mask[start_idx:end_idx]
        total_true = np.sum(window_mask)
        total_window_size = window_mask.size

        conf_vec[time_idx] = total_true / total_window_size # record confidence for current row

        # For observed values, confidence is 1; for imputed, it's the ratio of observed values to the window size
        for sensor_idx in range(num_sensors):
            if mask[time_idx, sensor_idx]:
                conf[time_idx, sensor_idx] = 1.0
            else:
                conf[time_idx, sensor_idx] = conf_vec[time_idx]

    return conf, conf_vec


def slice_up(imputed_obs, real_obs, W, stride, conf_vec):
    patches = []
    seq_len, num_sensors = imputed_obs.shape
    for time_idx in range(W, seq_len-W, stride): # slice center time index
        patches.append({
            'obs': imputed_obs[time_idx-W: time_idx+W+1],
            'real_obs': real_obs[time_idx-W: time_idx+W+1],
            'conf': conf_vec[time_idx],
            'idx': time_idx,
            'bin_id': None,
            'cluster_id': None,
            })
    return patches


def main():
    args = parse_args()
    print(f'Start imputing dataset {args.fp} using strategy: "{args.strat}"...')

    with open(args.fp, 'rb') as f:
        src_dataset = pickle.load(f)

    src_metadata = src_dataset['meta']
    src_segments = src_dataset['data']
    db_name = src_metadata['name']

    imp_segments = []
    imp_metadata = src_metadata.copy()

    mean = src_metadata['mean']
    num_sensors = src_metadata['num_sensors']
    # drop sensors if missing rate threshold < 100%
    if args.mrthold < 1.:
        sensor_mask = src_metadata['missing_rates'] <= args.mrthold
        print(f"Dropped sensors with high missing rates (>{args.mrthold:.2%})")
        num_sensors = np.sum(sensor_mask)
        print(f"Remaining number of sensors: {num_sensors}")
        drop_vars_in_metadata(imp_metadata, sensor_mask)

    if args.strat == 'spline': # TODO: not tested yet
        for i, src_segment in enumerate(tqdm(src_segments, desc="Processing segments...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            imp = impute_spline(obs, mean, mask=mask, stamp=stamp)
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })

    elif args.strat == 'linear': # TODO: not tested yet
        for i, src_segment in enumerate(tqdm(src_segments, desc="Processing segments...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            imp = impute_linear(obs, mean, mask=mask, stamp=stamp)
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })
        # imp_handler.set_metadata(meta_dict=src_metadata)
        
    elif args.strat == 'kmeans': # TODO: 1. not tested yet; 2. parameterization - impute test set with train set parameters
        # some constants
        config = KMeansImputerConfig()
        print(f"K-means imputer's config: {json.dumps(asdict(config))}")
        W = config.W
        stride = config.stride
        num_bins = config.num_bins
        num_kernels = config.num_kernels
        num_clusters = config.num_clusters

        all_patches = {}
        for i, src_segment in enumerate(tqdm(src_segments, desc="Slicing dataset into patches...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            obs_hat = impute_spline(obs, mean, mask=mask, stamp=stamp)
            conf, conf_vec = calculate_imputation_confidence_global_window(mask, W)
            patches = slice_up(obs_hat, obs, W, stride, conf_vec)
            all_patches[name] = patches
            # if i == 100:
            #     break
        
        print('Calculating histogram...')
        # Extracting all patches along with their name
        all_patches_flat = [(name, patch) for name, patches in all_patches.items() for patch in patches]
        # Sorting all patches by 'conf' (confidence)
        all_patches_sorted = sorted(all_patches_flat, key=lambda x: x[1]['conf'], reverse=True)
        # The length of all_patches_sorted will be used to determine how many items go into each bin
        total_patches = len(all_patches_sorted)
        # Calculating the number of items per bin, with the last bin potentially containing slightly more due to rounding
        items_per_bin = total_patches // num_bins
        # Creating bins for sorted patches
        bins = [all_patches_sorted[i*items_per_bin : (i+1)*items_per_bin] for i in range(num_bins-1)]
        # Adding the remaining items to the last bin, which includes the last set and any extras due to integer division
        bins.append(all_patches_sorted[(num_bins-1)*items_per_bin:]) # bin is sorted from high confidence to low
        
        print('Fitting Saab modules...')
        saab_modules = []
        for i in range(num_bins):
            saab_module = Saab(num_kernels=num_kernels)
            X = []
            for name, patch in bins[i]:
                x = patch['obs'] # (2W+1) by num_sensors
                # Flattening the array by concatenating each column
                x = x.flatten(order='F')
                X.append(x)
            X = np.vstack(X)
            saab_module.fit(X)
            saab_modules.append(saab_module)

        print('Fitting a k-means module...')

        print('Calculating temporal means over all sensors, in all clusters...')
        cluster_means = {i: {} for i in range(num_bins)}

        # do k-means clustering in each bin
        for i in range(num_bins):
            
            # Extract the transformed observations for k-means
            reduced_obs = [saab_modules[i].transform(patch['obs'].flatten(order='F')).squeeze() for name, patch in bins[i]]
            
            if reduced_obs:
                kmeans = KMeans(n_clusters=num_clusters, n_init='auto', init='random', random_state=0).fit(reduced_obs)
                labels = kmeans.labels_
                # print(labels)
                print(f"Bin {i}: {len(labels)} patches clustered into {num_clusters} clusters. Calculating mean over each...")
                
                # Grouping original observations by their cluster assignment
                clusters = {cluster_id: [] for cluster_id in range(num_clusters)}
                for j, (name, patch) in enumerate(bins[i]):
                    cluster_id = labels[j]
                    patch['cluster_id'] = cluster_id
                    patch['bin_id'] = i
                    clusters[cluster_id].append(patch['real_obs'])
                
                # Calculating the temporal mean for each cluster
                for cluster_id, obs_list in clusters.items():
                    # print(len(obs_list), obs_list[0].shape)
                    # Assuming `obs` is a 2D array where the mean should be calculated across all observations
                    # This calculates the mean across the 0th dimension, effectively averaging over all patches in the cluster
                    cluster_mean = np.nanmean(np.array(obs_list), axis=0)
                    # Final calculation: average across the first dimension of the mean patch
                    final_cluster_mean = np.nanmean(cluster_mean, axis=0)  # This results in (num_sensors,)
                    # Identify NaN positions in final_cluster_mean
                    nan_positions = np.isnan(final_cluster_mean)
                    # print(nan_positions)
                    # print(final_cluster_mean)
                    # print(mean)
                    # Replace NaNs in final_cluster_mean with the corresponding values from mean
                    final_cluster_mean[nan_positions] = mean[nan_positions]
                    cluster_means[i][cluster_id] = final_cluster_mean
            else:
                print(f"Bin {i}: No patches to cluster")

        for i, src_segment in enumerate(tqdm(src_segments, desc="Imputing each segment...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            obs_hat = impute_spline(obs, mean, mask=mask, stamp=stamp)
            conf, conf_vec = calculate_imputation_confidence_global_window(mask, W)
            imp = np.zeros_like(obs_hat, dtype=np.float32)
            imp += conf * obs_hat
            seq_len = obs.shape[0]
            for j in range(seq_len):
                time_idx = j
                if j < W:
                    time_idx = W
                elif j > seq_len - W - 1:
                    time_idx = seq_len - W - 1
                # print(len(all_patches[name]))
                patch = all_patches[name][time_idx-W]
                cluster_mean = cluster_means[patch['bin_id']][patch['cluster_id']]
                imp[j] += (1-conf)[j] * cluster_mean
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })
            # if i == 100:
            #     break
        # imp_handler.set_metadata(meta_dict=src_metadata)

    elif args.strat == 'xgboost': # TODO: not tested yet
        # some constants
        config = XGBoostImputerConfig()
        print(f"XGBoost imputer's config: {json.dumps(asdict(config))}")
        W = config.W

        all_segments = {}
        all_patches = []
        
        for i, src_segment in enumerate(tqdm(src_segments, desc="Slicing dataset into patches...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']

            # drop features columns with high missing rates:
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]

            # obs_hat = impute_spline(obs, mean, mask=mask, stamp=stamp)
            # conf, conf_vec = calculate_imputation_confidence_global_window(mask, W)
            seq_len = len(stamp)
            for time_idx in range(W, seq_len-W): # slice center time index
                all_patches.append({
                    'name': name,
                    'obs': obs[time_idx-W: time_idx+W+1],
                    # 'obs_hat': obs_hat[time_idx-W: time_idx+W+1],
                    # 'conf': conf_vec[time_idx],
                    'idx': time_idx,
                    })
            all_segments[name] = {
                'obs': obs,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            }
            # if i == 100:
            #     break

        models = []

        for s in tqdm(range(num_sensors), desc="Training models per sensor..."): # for each sensor, train a XGBoost for imputation
            X = [] # feature list, each feature is a 1-D array
            Y = [] # target list, each target is also a 1-D array, or a scalar

            for patch in all_patches:
                obs = patch['obs']
                if not np.isnan(obs[W][s]): # this patch can be used to train the model
                    obs_copy = obs.copy()
                    target = obs_copy[W][s]
                    obs_copy[W][s] = np.nan
                    obs_flatten = obs_copy.ravel() # flatten always return a copy, ravel returns a view wherever possible
                    # print(obs_flatten[W*num_sensors+s]) # make sure the target is set to nan
                    X.append(obs_flatten)
                    Y.append(target)
            
            X = np.array(X)
            Y = np.array(Y)
            # print(X.shape)
            # print(Y.shape)

            # Define XGBoost model
            model = xgb.XGBRegressor(objective='reg:squarederror')

            # Train the model
            model.fit(X, Y)

            models.append(model)

            # predictions = model.predict(X)

            # print("Predictions:", predictions)
            # print("Ground truths:", Y)

        # interpolate with xgboost
        for patch in tqdm(all_patches, desc="Interpolating with XGBoost..."):
            obs = patch['obs']
            name = patch['name']
            idx = patch['idx']
            for s in range(num_sensors):
                if np.isnan(obs[W][s]): # this position needs to be interpolated
                    src_segment = all_segments[name]
                    src_segment['obs'][idx][s] = models[s].predict(np.array([obs.flatten()]))[0]
        # interpolate head and tail with spline
        for name, src_segment in tqdm(all_segments.items(), desc="Writing results..."):
            obs = src_segment['obs']
            mask = src_segment['mask']
            stamp = src_segment['stamp']
            label = src_segment['label']
            demogr = src_segment['demogr']
            obs = impute_forward_fill(obs, mean) # no mask, so impute_spline calculates mask using obs
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })

    elif args.strat == 'zero':
        for i, src_segment in enumerate(tqdm(src_segments, desc="Processing segments...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            imp = impute_all_zero(obs, mask=mask)
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })

    elif args.strat == 'mean':
        for i, src_segment in enumerate(tqdm(src_segments, desc="Processing segments...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            imp = impute_mean(obs, mean=mean, mask=mask)
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })

    elif args.strat == 'forward':
        for i, src_segment in enumerate(tqdm(src_segments, desc="Processing segments...")):
            name = src_segment['name']
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            if args.mrthold < 1.:
                obs = obs[:,sensor_mask]
                mask = mask[:,sensor_mask]
            imp = impute_forward_fill(obs, mean=mean)
            imp_segments.append({
                'name': name,
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })

    else: # TODO: zero, mean and forward
        raise NotImplementedError()

    # save the imputed dataset to a pickle file
    output_fp = os.path.join(args.out_dir, db_name+'_imputed_'+args.strat+'.pkl')
    with open(output_fp, 'wb') as f:
        pickle.dump({
            'data': imp_segments,
            'meta': imp_metadata,
        }, f)
    
    print(f'Imputed dataset has been saved as {output_fp}')
    print_metadata(imp_metadata)

if __name__ == '__main__':
    main()