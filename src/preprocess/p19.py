"""
Preprocess PhysioNet Challenge 2019 Sepsis Early Prediction (P19) dataset.

The purpose of preprocessing:
    - Getting rid of outliers.
    - Resampling (if needed). P19 is originally sampled at a fixed interval (1hr) so resampling is not necessary.
    - Computing statistics (mean and std) over the whole dataset and recording them for later use (normalizing, etc.).
    - Converting categorical features (observations) into numeric (integer) labels.

This program takes 3 keyword arguments:
    - `root_dir`: Path to the original P19 data's root directory (`training/`) that contains both `training_setA` and `training_setB`.
    - `resample`: Resampling interval in hours. Default is None (no resampling). P19 is originally sampled at a 1-hour interval, so resampling is generally unnecessary. Code doesn't handle resampling at all for now.
    - `out_dir`: Path to the output directory.

Output:
    - A pickle file will be saved to `out_dir/P19.pkl`, containing a dictionary with two keys: `data` and `meta`.

The pickle file structure:
    - `data` (list of dictionaries): Each dictionary represents a time series segment (episode) with the following key-value pairs:
        - `obs` (ndarray of float32): Raw (non-normalized, categorical features are replaced by integer labels if any) sensor observation array.
            - Shape: (sequence_length, num_sensors).
            - Missing values are NaNs.
        - `stamp` (ndarray of float32): Timestamps of the observations. Shape: (sequence_length).
        - `mask` (ndarray of boolean): Missing observation mask, where `True` indicates a real observed value, and `False` indicates a missing value (which may be imputed later). Same shape as `obs`.
        - `label` (int): Classification label of the time series.
        - `demogr` (dictionary): Demographic information containing the following keys:
            - `age` (int): Age of the patient, with 100 representing patients aged 90 or above.
            - `gender` (int): 0 for female and 1 for male.
            - `unit_id` (int): 0 for MICU, 1 for SICU, and -1 for unknown.
            - `hosp_adm_time` (float): Time difference between ICU admission and hospital admission, in hours (usually negative).
            - `icu_los` (float): Length of ICU stay in hours.

    - `meta` (dictionary): Metadata providing information about the processed dataset, including:
        - `size` (int): Number of segments in the dataset.
        - `num_sensors` (int): Number of sensors (feature columns).
        - `num_classes` (int): Number of classification classes.
        - `sensor_names` (list of strings): Names of the sensors, sized `(num_sensors)`.
        - `class_names` (list of strings): Names of the classes, from 0 to `num_classes-1`.
        - `class_counts` (ndarray of int32): Number of samples in each class, sized `(num_classes)`.
        - `mean` (ndarray of float32): Mean observed values for each sensor, shaped as `(num_sensors)`.
        - `std` (ndarray of float32): Standard deviation of observed values for each sensor, shaped as `(num_sensors)`.
        - `missing_rates` (ndarray of float32): Missing rates for each sensor modality, shaped as `(num_sensors)`.
        - `is_categorical` (ndarray of boolean): Indicates if there are any categorical columns (sensors).
        - `sample_interval` (float32): Dataset's constant sample interval in hours. P19 is originally sampled at 1.0 hr.
            - If this value is negative or NaN, observations may not be regularly sampled, meaning consecutive rows may not have consistent time intervals.
            - In such cases, `stamp` can be meaningful. Otherwise, `stamp` is redundant and can be inferred by multiplying the row index by the sample rate (in hours).
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
from glob import glob
from tqdm import tqdm
from tabulate import tabulate

from ..utils import print_metadata


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess P19.')
    parser.add_argument('--root_dir', type=str, default='./raw_data/P19/training/',
                        help='Path to the original P19 data\'s root directory that contains both training_setA and training_setB.')
    parser.add_argument('--resample', type=float, default=None,
                        help='Resampling interval in hour. Default is None (no-resampling).')  # useless for now
    parser.add_argument('--out_dir', type=str, default='./data/',
                        help='Path to the directory where the output will be stored.')
    args = parser.parse_args()
    return args


def process_psv(fp):
    df = pd.read_csv(fp, sep='|')
    # extract demographic information
    if pd.isna(df.iloc[0]['Unit1']) or pd.isna(df.iloc[0]['Unit2']):
        unit_id = -1
    else:
        unit_id = int(df.iloc[0]['Unit2'])  # 0 for unit1 (micu), 1 for unit2 (sicu)
        # check unit1 and unit2 can't be both 0 or 1:
        assert int(df.iloc[0]['Unit1']) + int(df.iloc[0]['Unit2']) == 1
    demogr = {
        'age': int(df.iloc[0]['Age']),
        'gender': int(df.iloc[0]['Gender']),
        'unit_id': unit_id,
        'hosp_adm_time': float(df.iloc[0]['HospAdmTime']),
        'icu_los': float(df['ICULOS'].iloc[-1])  # last value as it represents the length
    }
    obs = df.iloc[:, :34].values.astype(np.float32)  # first 34 columns are observations
    mask = ~df.iloc[:, :34].isna().values  # inverse of isna() for mask
    # ICULOS (ICU length of stay) as timestamps
    stamp = df['ICULOS'].values.astype(np.float32)
    # extract sepsis label from the last row
    label = int(df['SepsisLabel'].iloc[-1])
    return obs, stamp, mask, label, demogr


def main():
    args = parse_args()

    # define pickle file path
    pkl_file_path = os.path.join(args.out_dir, 'P19.pkl')
    os.makedirs(args.out_dir, exist_ok=True)

    psv_paths = glob(os.path.join(args.root_dir, 'training_set*/p*.psv'))
    print(f'Found {len(psv_paths)} psv files. Start processing.')

    # accumulators for metadata
    sum_obs = np.zeros(34, dtype=np.float64)
    sum_sq_obs = np.zeros(34, dtype=np.float64)
    count_obs = np.zeros(34, dtype=np.float64)
    count_nan_obs = np.zeros(34, dtype=np.float64)
    seq_lens = []

    # initialize metadata records
    metadata = {
        'name': 'P19',
        'size': 0,  # to be counted later on the fly
        'num_sensors': 34,  # first 34 columns are sensors
        'num_classes': 2,  # binary classification for sepsis
        'sensor_names': [],  # to be filled later
        'class_names': ['Non-sepsis', 'Sepsis'],
        'class_counts': np.zeros(2, dtype=np.int32),  # to be counted later on the fly
        'mean': None,  # to be filled later
        'std': None,  # to be filled later
        'missing_rates': None,  # to be filled later
        'is_categorical': np.array([False] * 34),  # no categorical feature in P19
        'sample_interval': args.resample if args.resample is not None and args.resample > 0 else np.nan,
        'avg_seq_len': None,  # to be filled later
        'max_seq_len': None,  # to be filled later
        'min_seq_len': None,  # to be filled later
        'seq_len_sigma': None,  # to be filled later
    }

    # get column names
    example_df = pd.read_csv(psv_paths[0], sep='|')
    metadata['sensor_names'] = example_df.columns.tolist()[:34]

    # initialize the data structure for the pickle file
    ds_dict = {
        'meta': metadata,
        'data': [],
    }

    for i, fp in enumerate(tqdm(psv_paths, desc="Processing .psv files...")):
        try:
            obs, stamp, mask, label, demogr = process_psv(fp)
        except Exception as e:
            print(f'Something is off ({type(e)}) processing {fp}, skipping it.')
            continue

        seq_lens.append(len(stamp))

        # prepare the data dictionary for each file
        segment_data = {
            'name': os.path.splitext(os.path.basename(fp))[0],
            'obs': obs,
            'stamp': stamp,
            'mask': mask,
            'label': label,
            'demogr': demogr,
        }

        # add the segment data to the dataset
        ds_dict['data'].append(segment_data)

        # update metadata
        metadata['size'] += 1
        metadata['class_counts'][label] += 1

        # update sums for mean/std calculation
        valid_obs_mask = ~np.isnan(obs)
        sum_obs += np.where(valid_obs_mask, obs, 0.).sum(axis=0)
        sum_sq_obs += np.where(valid_obs_mask, obs**2, 0.).sum(axis=0)
        count_obs += valid_obs_mask.sum(axis=0)

        # count missing values
        nan_obs_mask = np.isnan(obs)
        count_nan_obs += nan_obs_mask.sum(axis=0)

    # calculate mean and std
    mean_obs = sum_obs / count_obs
    var_obs = (sum_sq_obs - sum_obs**2 / count_obs) / count_obs
    std_obs = np.sqrt(var_obs)

    # calculate missing rates
    total_obs = count_obs + count_nan_obs
    missing_rates = count_nan_obs / total_obs

    # update metadata with calculated mean, std, and missing rates
    metadata['mean'] = mean_obs.astype(np.float32)
    metadata['std'] = std_obs.astype(np.float32)
    metadata['missing_rates'] = missing_rates.astype(np.float32)

    metadata['avg_seq_len'] = np.mean(seq_lens)
    metadata['max_seq_len'] = np.max(seq_lens)
    metadata['min_seq_len'] = np.min(seq_lens)
    metadata['seq_len_sigma'] = np.std(seq_lens)

    # save the dataset to a pickle file
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(ds_dict, f)

    print(f'Processed {metadata["size"]} files and dataset saved to {pkl_file_path}.')

    print_metadata(metadata)


if __name__ == '__main__':
    main()