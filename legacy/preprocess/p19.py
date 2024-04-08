"""
Preprocess PhysioNet Challenge 2019 Sepsis Early Prediction (P19) dataset.
What preprocessing actually does:
    Getting rid of outliers.
    Resampling (if needed). P19 is originally sampled at a fixed interval (1hr) so resampling is not necessary.
    Computing statistics (mean and std) over the whole dataset and record them for later use (normalizing etc.).
    Converting categorical features (observations) into numeric (integer) labels.
This program takes 2 keyword arguments:
    `root_dir`: path to the original P19 data's root directory (`training\`) that contains both `training_setA` and `training_setB`.
    `resample`: resampling interval in hour. Default is 1.0 (P19 is itself sampled at 1hr interval so basically no resampling).
    `out_dir`: path to the output directory.
        All time series episodes will be saved under `out_dir/segments/` as individual pickles (a single dictionary) with 4 keys in each:
            `obs` (ndarray of float32): raw (non-normalized, categorical features are simply replaced as integer labels if any) sensor observation array.
                Shaped as (len, num_sensors).
                Missing values are NaNs.
            `stamp` (ndarray of float32): time stamps of size (len).
            `mask` (ndarray of boolean): missing observation masking, 1 for real observed value and 0 for missing value (maybe imputed later).
                Same shape as `obs`.
            `label` (int): classification label of the time series.
            `demogr` (dict): dictionary of demographic info. In P19 there are 5 keys:
                `age` (int): 100 for patients aged 90 or above.
                `gender` (int): 0 for female and 1 for male.
                `unit_id` (int / None): 0 for micu, 1 for sicu and None for unknown.
                `hosp_adm_time` (float): t(icu admission) - t(hospital admission) in hours, usually negative.
                `icu_los` (float): length of icu stay in hours.
        A metafile `meta.pkl` (essentially a dictionary) will also be saved to this directory, providing some meta info about the processed dataset such as:
            `size` (int): size of the dataset.
            `num_sensors` (int): how many sensors (feature columns) are there.
            `num_classes` (int): how many classification classes are there.
            `sensor_names` (list of strings): names of the sensors, sized (num_sensors).
            `class_names` (list of strings): names of the classes, starting from 0 to num_classes-1.
            `mean` (ndarray of float32): mean observed values in each sensor, shaped as (num_sensors).
            `std` (ndarray of float32): std of observed values in each sensor, shaped as [num_sensors].
            `is_categorical` (ndarray of boolean): True if there is any categorical column (sensor).
            `resampled` (float32): dataset's sample rate. Unit is hour. Note that P19 is originally sampled at 1.0 hr.
                In case this term is None or negative, observations may not be regularly sampled, i.e. consecutive rows may not have same time intervals in between them.
                In such case, `stamp` can be meaningful.
                Otherwise, `stamp` is basically redundant and can simply be recovered by multiplying row index with sample rate (hr).
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import h5py
from glob import glob
from tqdm import tqdm

WRITE_OUT = True

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess P19.')
    parser.add_argument('--root_dir', type=str, default='../../raw_data/P19/training/',
                        help='Path to the original P19 data\'s root directory that contains both training_setA and training_setB.')
    parser.add_argument('--resample', type=float, default=None,
                        help='Resampling interval in hour. Default is None (no-resampling).') # useless for now
    parser.add_argument('--out_dir', type=str, default='../../data/',
                        help='Path to the directory where the output will be stored.')
    args = parser.parse_args()
    return args


def process_file(fp):
    df = pd.read_csv(fp, sep='|')
    # extract demographic information
    if pd.isna(df.iloc[0]['Unit1']) or pd.isna(df.iloc[0]['Unit2']):
        unit_id = -1
    else:
        unit_id = int(df.iloc[0]['Unit2']) # 0 for unit1 (micu), 1 for unit2 (sicu)
        # check unit1 and unit2 cant be both 0 or 1:
        assert int(df.iloc[0]['Unit1']) + int(df.iloc[0]['Unit2']) == 1
    demogr = {
        'age': int(df.iloc[0]['Age']),
        'gender': int(df.iloc[0]['Gender']),
        'unit_id': unit_id,
        'hosp_adm_time': float(df.iloc[0]['HospAdmTime']),
        'icu_los': float(df['ICULOS'].iloc[-1]) # last value as it represents the length
    }
    obs = df.iloc[:, :34].values.astype(np.float32)  # first 34 columns are observations
    mask = ~df.iloc[:, :34].isna().values  # inverse of isna() for mask
    # ICULOS as time stamps
    stamp = df['ICULOS'].values.astype(np.float32)
    # extract sepsis label from the last row
    label = int(df['SepsisLabel'].iloc[-1])
    return obs, stamp, mask, label, demogr


def main():
    args = parse_args()

    # ensure output directories exist
    segment_dir = os.path.join(args.out_dir, 'segments')
    if WRITE_OUT:
        os.makedirs(segment_dir, exist_ok=True)

    psv_paths = glob(os.path.join(args.root_dir, 'training_set*/p*.psv'))
    print(f'Found {len(psv_paths)} psv files. Start processing.')

    # accumulators for calculating mean and std
    sum_obs = np.zeros(34, dtype=np.float64)
    sum_sq_obs = np.zeros(34, dtype=np.float64)
    count_obs = np.zeros(34, dtype=np.float64)

    # init metadata records
    metadata = {
        'size': 0,
        'num_sensors': 34,  # first 34 columns are sensors
        'num_classes': 2,  # binary classification for sepsis
        'sensor_names': [],  # to be filled later
        'class_names': ['Non-sepsis', 'Sepsis'],
        'mean': [], # to be filled later
        'std': [], # to be filled later
        'is_categorical': np.array([False] * 34),  # no categorical feature in P19
        'sample_interval': args.resample
    }

    # get column names
    example_df = pd.read_csv(psv_paths[0], sep='|')
    metadata['sensor_names'] = example_df.columns.tolist()[:34]

    for i, fp in enumerate(tqdm(psv_paths, desc="Processing files")):
        try:
            obs, stamp, mask, label, demogr = process_file(fp)
        except Exception as e:
            print(f'Something is off ({type(e)}) processing {fp}, skipping it.')
            continue
        
        # update sum and sum of squares for non-missing values
        valid_obs_mask = ~np.isnan(obs)
        sum_obs += np.where(valid_obs_mask, obs, 0).sum(axis=0)
        sum_sq_obs += np.where(valid_obs_mask, obs**2, 0).sum(axis=0)
        count_obs += valid_obs_mask.sum(axis=0)
        
        # save current segment
        # extract the base name without the '.psv' extension
        base_name = os.path.basename(fp)
        segment_name = base_name.replace('.psv', '.pkl')
        # construct the path for the processed segment file
        segment_path = os.path.join(segment_dir, segment_name)
        if WRITE_OUT:
            with open(segment_path, 'wb') as f:
                pickle.dump({'obs': obs, 'stamp': stamp, 'mask': mask, 'label': label, 'demogr': demogr}, f)
            
        metadata['size'] += 1

    # calculate mean and std
    mean_obs = sum_obs / count_obs
    var_obs = (sum_sq_obs - sum_obs**2 / count_obs) / count_obs
    std_obs = np.sqrt(var_obs)

    # update metadata with calculated mean and std
    metadata['mean'] = mean_obs.astype(np.float32)
    metadata['std'] = std_obs.astype(np.float32)

    # save metadata
    if WRITE_OUT:
        with open(os.path.join(args.out_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

    print(f'Processed {metadata["size"]} files.')


if __name__ == '__main__':
    main()