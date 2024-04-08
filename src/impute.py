"""
Impute a preprocessed multivariate time sereis (missing values included) dataset by a specific strategy.
Keyword arguments are:
    `fp`: input hdf5 file path.
    `strat`: imputation strategy.

"""


import pandas as pd
import numpy as np
import pickle
import argparse
import os
import json
import h5py
from glob import glob
from tqdm import tqdm
from dataclasses import dataclass, asdict
import numpy as np
from scipy import interpolate
from sklearn.cluster import KMeans


from dataset import MyHDF5Handler
from pixelhop.saab import Saab


def parse_args():
    parser = argparse.ArgumentParser(description='Impute a preprocessed multivariate time series (missing values included) dataset.')
    parser.add_argument('--fp', type=str, default='../../data/',
                        help='Path to the processed dataset hdf5 file.')
    parser.add_argument('--strat', choices=['zero', 'mean', 'forward', 'linear', 'spline', 'ours'],
                        help='Choose imputation strategy.')
    parser.add_argument('--out_dir', type=str, default='../../data/',
                        help='Path to the directory where the output will be stored.')
    args = parser.parse_args()
    return args


@dataclass
class OurImputerConfig:
    W: int = 3
    stride: int = 1
    num_bins: int = 4
    num_kernels: int = 16
    num_clusters: int = 16

def impute_all_zero(obs, mask):
    """Impute all NaNs with 0."""
    imp = np.copy(obs)
    imp[~mask] = 0
    return imp

def impute_mean(obs, mask, mean):
    """Impute all NaNs with the mean value of their respective sensor."""
    imp = np.copy(obs)
    for i in range(obs.shape[1]):
        imp[~mask[:, i], i] = mean[i]
    return imp

def impute_forward_fill(obs, mask, mean):
    """Impute starting NaNs with mean, then forward fill the rest."""
    imp = np.copy(obs)
    for i in range(obs.shape[1]):
        if np.isnan(imp[0, i]):
            imp[0, i] = mean[i]
        for j in range(1, len(imp)):
            if np.isnan(imp[j, i]):
                imp[j, i] = imp[j-1, i]
    return imp

def impute_linear(obs, mask, mean, stamp=None):
    """Linear interpolation with time stamps and conditions for all-NaN columns or single non-NaN values."""
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

def impute_spline(obs, mask, mean, stamp=None):
    """Cubic spline interpolation with time stamps for smoother imputation."""
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
    src_handler = MyHDF5Handler(args.fp, read_only=True) # source dataset
    # src_handler.print_metadata()

    src_metadata = src_handler.get_metadata()
    size = src_metadata['size']

    db_name, db_ext = os.path.splitext(os.path.basename(args.fp))

    imp_handler = MyHDF5Handler(os.path.join(args.out_dir, db_name+'_'+args.strat+db_ext), read_only=False)

    all_segment_keys = src_handler.get_all_segment_keys()

    if args.strat == 'spline':
        mean = src_metadata['mean']
        # for segment_key in all_segment_keys:
        for i, segment_key in enumerate(tqdm(all_segment_keys, desc="Processing segments...")):
            src_segment = src_handler.get_segment(segment_key)
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            imp = impute_spline(obs, mask, mean, stamp=stamp)
            imp_handler.add_segment(segment_key, {
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })
        imp_handler.set_metadata(meta_dict=src_metadata)
        
    elif args.strat == 'ours':
        # some constants
        config = OurImputerConfig()
        print(f"Our imputer's config: {json.dumps(asdict(config))}")
        W = config.W
        stride = config.stride
        num_bins = config.num_bins
        num_kernels = config.num_kernels
        num_clusters = config.num_clusters

        mean = src_metadata['mean']
        all_patches = {}
        # for segment_key in all_segment_keys:
        for i, segment_key in enumerate(tqdm(all_segment_keys, desc='Slicing up...')):
            src_segment = src_handler.get_segment(segment_key)
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            obs_hat = impute_spline(obs, mask, mean, stamp=stamp)
            conf, conf_vec = calculate_imputation_confidence_global_window(mask, W)
            patches = slice_up(obs_hat, obs, W, stride, conf_vec)
            all_patches[segment_key] = patches
            # if i == 100:
            #     break
        
        print('Calculating histogram...')
        # Extracting all patches along with their segment_key
        all_patches_flat = [(segment_key, patch) for segment_key, patches in all_patches.items() for patch in patches]
        # Sorting all patches by 'conf'
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
            for segment_key, patch in bins[i]:
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
        for i in range(num_bins):
            
            # Extract the transformed observations for k-means
            reduced_obs = [saab_modules[i].transform(patch['obs'].flatten(order='F')).squeeze() for segment_key, patch in bins[i]]
            
            if reduced_obs:
                kmeans = KMeans(n_clusters=num_clusters, n_init='auto', init='random', random_state=0).fit(reduced_obs)
                labels = kmeans.labels_
                # print(labels)
                print(f"Bin {i}: {len(labels)} patches clustered into {num_clusters} clusters. Calculating mean over each...")
                
                # Grouping original observations by their cluster assignment
                clusters = {cluster_id: [] for cluster_id in range(num_clusters)}
                for j, (segment_key, patch) in enumerate(bins[i]):
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

        for i, segment_key in enumerate(tqdm(all_segment_keys, desc="Imputing...")):
            src_segment = src_handler.get_segment(segment_key)
            obs = src_segment['obs']
            stamp = src_segment['stamp']
            mask = src_segment['mask']
            label = src_segment['label']
            demogr = src_segment['demogr']
            obs_hat = impute_spline(obs, mask, mean, stamp=stamp)
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
                # print(len(all_patches[segment_key]))
                patch = all_patches[segment_key][time_idx-W]
                cluster_mean = cluster_means[patch['bin_id']][patch['cluster_id']]
                imp[j] += (1-conf)[j] * cluster_mean
            imp_handler.add_segment(segment_key, {
                'obs': imp,
                'stamp': stamp,
                'mask': mask,
                'label': label,
                'demogr': demogr,
            })
            # if i == 100:
            #     break
        imp_handler.set_metadata(meta_dict=src_metadata)

    else:
        raise NotImplementedError()

if __name__ == '__main__':
    main()