"""
The PhysioNet/Computing in Cardiology Challenge 2012 Dataset - Preprocessing Script.

The purpose of preprocessing:
    - Getting rid of outliers.
    - Resampling (if needed).
    - Computing meta statistics (mean and std) over the whole dataset and recording them for later use (normalizing, etc.).
    - Converting categorical features (observations) into numeric (integer) labels.
    - Splitting the dataset into train/valid/test.

This program takes 3 keyword arguments:
    - `root_dir`: Path to the original P12 data's root directory that contains `set-a`, `set-b` and `set-c`.
    - `resample`: Resampling interval in hours. Default is None (no resampling).
    - `out_dir`: Path to the output directory.

Output:
    - A pickle file will be saved to `out_dir/p12.pkl`, containing a dictionary with two keys: `data` and `meta`.

The pickle file structure:
    - `data` (list of dictionaries): Each dictionary represents a time series segment (episode) with the following key-value pairs:
        - `var` (ndarray of float32): Raw (non-normalized, categorical features are replaced by integer labels if any) var observation array.
            - Shape: (seq_len, num_vars).
            - Missing values are NaNs.
        - `time` (ndarray of float32): Timestamps of the observations. Shape: (seq_len).
        - `interval` (ndarray of float32): Time interval since last observation. Shape: (seq_len, num_vars).
        - `mask` (ndarray of boolean): Missing observation mask, where `True` indicates a real observed value, and `False` indicates a missing value (which may be imputed later). Same shape as `obs`.
        - `label` (ndarray of float32): Label of the time series. Shape: (num_labels)
        - `static` (dictionary): Static, usually demographic information.

    - `meta` (dictionary): Metadata providing information about the processed dataset, including:
        - `name` (str): Name of dataset.
        - `split` (str): Split of dataset (whole/train/eval/test).
        - `desc` (any): Just a description, you can put any info here.
        - `time_unit` (str): Time unit (hours/minutes/seconds/...).
        - `impute_strat` (str): Name of the imputer, None if not imputed.
        - `size` (int): Number of segments in the dataset.
        - `num_vars` (int): Number of vars (feature columns).
        - `var_names` (list of strings): Names of the vars, sized `(num_vars)`.
        - `mean` (ndarray of float32): Mean observed values for each variable, shaped as `(num_vars)`.
        - `std` (ndarray of float32): Standard deviation of observed values for each variable, shaped as `(num_vars)`.
        - `missing_rates` (ndarray of float32): Missing rates for each variable, shaped as `(num_vars)`.
        - `is_categorical` (ndarray of boolean): Indicates if there are any categorical columns (vars).
        - `sample_interval` (float32): Dataset's constant sample interval in hours. P19 is originally sampled at 1.0 hr.
            - If this value is negative or NaN or None, observations may not be regularly sampled, meaning consecutive rows may not have consistent time intervals.
            - In such cases, `stamp` can be meaningful. Otherwise, `stamp` is redundant and can be inferred by multiplying the row index by the sample rate (in hours).
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import pickle
import random
from ..utils import print_metadata
    
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
# rest will be test set

args = None
def parse_args():
    global args
    parser = argparse.ArgumentParser(description='Preprocess P12.')
    parser.add_argument('--root-dir', type=str, default='./raw_data/p12/',
                        help='Path to the original P12 data\'s root directory that contains folders set-a, set-b and set-c and the outcomes.')
    parser.add_argument('--resample', type=float, default=None,
                        help='Resampling interval in hour. Default is None (no-resampling).')  # useless for now
    parser.add_argument('--out-dir', type=str, default='./data/p12',
                        help='Path to the directory where the output will be stored.')
    parser.add_argument('--do-split', action='store_true', default=False,
                        help='Split data into a train set, a validation set and a test test; ratio is configurable in source code.')
    args = parser.parse_args()
    return args


# >>> descriptor field names and time series variable field names >>>
# # from P12 official webpage: https://www.physionet.org/content/challenge-2012/1.0.0/
descriptors = [
    "RecordID", "Age", "Gender", "Height", "ICUType", "Weight" # note that 'Weight' is also a time series variable
]

variables = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine", 
    "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", 
    "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", 
    "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", 
    "WBC", "Weight"
]

label_names = ['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival', 'In-hospital_death']
# <<< descriptor field names and time series variable field names <<<


def hh_mm_to_hours(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours + minutes / 60


def process_split(split_name: str, raw_data_list: list):
    data_list = []
    # initialize metadata records
    metadata = {
        'name': 'p12',
        'desc': 'Cleaned PhysioNet Challenge 2012 dataset.',
        'split': split_name,
        'time_unit': 'hour',
        'impute_strat': None,
        'size': 0,  # to be counted later on the fly
        'num_vars': len(variables),
        'var_names': variables,
        'mean': None,  # to be filled later
        'std': None,  # to be filled later
        'missing_rates': None,  # to be filled later
        'is_categorical': np.array([False] * len(variables)),  # no categorical feature in P12
        'sample_interval': args.resample if args.resample is not None and args.resample > 0 else None,
        'mean_seq_len': None,  # to be filled later
        'max_seq_len': None,  # to be filled later
        'min_seq_len': None,  # to be filled later
        'sigma_seq_len': None,  # to be filled later
    }

    # accumulators
    sum_obs = np.zeros(metadata['num_vars'], dtype=np.float64)
    sum_sq_obs = np.zeros(metadata['num_vars'], dtype=np.float64)
    count_obs = np.zeros(metadata['num_vars'], dtype=np.float64)
    count_nan_obs = np.zeros(metadata['num_vars'], dtype=np.float64)
    seq_lens = []

    # remove blacklist patients
    blacklist = set(['140501', '150649', '140936', '143656', '141264', '145611', '142998', '147514', '142731', '150309', '155655', '156254'])

    for i, data_dict in tqdm(enumerate(raw_data_list), desc="Post processing..."):

        # >>> post-processing, remove outliers >>>
        if data_dict['id'] in blacklist:
            print(f"Blacklisted sample id {data_dict['id']}: passed")
            continue
        # <<< post-processing, remove outliers <<<

        data_list.append(data_dict) # all data entering here are considered valid

        seq_lens.append(len(data_dict['time']))

        # update dataset size
        metadata['size'] += 1

        # update sums for mean/std calculation
        valid_obs_mask = ~np.isnan(data_dict['var'])
        sum_obs += np.where(valid_obs_mask, data_dict['var'], 0.).sum(axis=0)
        sum_sq_obs += np.where(valid_obs_mask, data_dict['var']**2, 0.).sum(axis=0)
        count_obs += valid_obs_mask.sum(axis=0)

        # count missing values
        nan_obs_mask = np.isnan(data_dict['var'])
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

    metadata['mean_seq_len'] = np.mean(seq_lens)
    metadata['max_seq_len'] = np.max(seq_lens)
    metadata['min_seq_len'] = np.min(seq_lens)
    metadata['sigma_seq_len'] = np.std(seq_lens)

    return data_list, metadata


def save_pkl(filename, data_list, metadata):
    # define pickle file path
    pkl_file_path = os.path.join(args.out_dir, filename)
    os.makedirs(args.out_dir, exist_ok=True)

    # save the dataset to a pickle file
    with open(pkl_file_path, 'wb') as f:
        pickle.dump({
            'data': data_list,
            'meta': metadata,
        }, f)

    print(f'Processed {metadata["size"]} valid records; dataset saved to {pkl_file_path}.')
    print_metadata(metadata)


def main():
    args = parse_args()
    root_dir = args.root_dir
    print("Collecting labels...")
    outcome_files = ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']
    all_labels = {}

    for file in outcome_files:
        file_path = os.path.join(root_dir, file)

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            record_id = str(row['RecordID']) # make sure id is str
            row.drop('RecordID', inplace=True)
            all_labels[record_id] = row.to_dict()

    sub_dirs = ['set-a', 'set-b', 'set-c']
    raw_data_list = []
    cnt = 0
    all_files = []
    for sub_dir in sub_dirs:
        all_files.extend(glob(os.path.join(root_dir, sub_dir, '*.txt')))
    all_files.sort()
    for f in tqdm(all_files, desc='Parsing CSV files...'):
        id, _ = os.path.splitext(os.path.basename(f))
        if id not in all_labels:
            continue
        df = pd.read_csv(f, sep=",", header=1, names=["time", "param", "value"])
        df_static = df.iloc[0:5] # including weight on admission
        df_var = df.iloc[4:] # also including weight on admission
        if df_var.iloc[0]['param'] == 'Weight' and df_var.iloc[0]['value'] < 0:
            df_var = df_var.drop(index=4)  # drop the first row if weight on admission is not measured
        demogr = dict(zip(df_static['param'], df_static['value'])) # demographic info dictionary
        df_var = df_var.pivot_table(index='time', columns='param', values='value', aggfunc='mean')
        # reindex to list `variables` to ensure all 37 variables are present, even if they are missing in the data
        df_var = df_var.reindex(columns=variables)
        df_var = df_var.reset_index()

        # add a column that convert time 'hh:mm' string into hours in float
        df_var['hour'] = df_var['time'].apply(hh_mm_to_hours)
        # extract the 'time' vector and the 'var' array
        time = df_var['hour'].values.astype(np.float32)
        var = df_var.drop(columns=['time', 'hour']).to_numpy(dtype=np.float32)

        mask = np.isnan(var) # same definition as in GRU-D, 1 for observation 0 for missingness
        interval = np.zeros_like(var)
        for col in range(var.shape[1]):
            last_observed_time = 0
            for row in range(var.shape[0]):
                interval[row, col] = time[row] - last_observed_time
                if mask[row, col]: # observed here
                    last_observed_time = time[row]
        data_dict = {
            'id': id,
            'time': time,
            'var': var,
            'mask': mask,
            'interval': interval,
            'static': demogr,
            'label': all_labels[id],
        }
        raw_data_list.append(data_dict)

    if args.do_split:
        print(f"Dataset split ratio (train:valid:test)= {TRAIN_RATIO}:{VALID_RATIO}:{1-TRAIN_RATIO-VALID_RATIO}, shuffling data...")

        random.shuffle(raw_data_list)
        # calculate split indices
        train_size = int(TRAIN_RATIO * len(raw_data_list))
        valid_size = int(VALID_RATIO * len(raw_data_list))
        test_size = len(raw_data_list) - train_size - valid_size

        # split the data
        raw_data_list_splits = {
            'train': raw_data_list[:train_size],
            'valid': raw_data_list[train_size:train_size + valid_size],
            'test': raw_data_list[train_size + valid_size:],
        }

        for split_name in ['train', 'valid', 'test']:
            data_list, metadata = process_split(split_name, raw_data_list_splits[split_name])
            save_pkl('p12_clean_'+split_name+'.pkl', data_list, metadata)
        

    else:
        data_list, metadata = process_split('whole', raw_data_list)
        save_pkl('p12_clean_whole.pkl', data_list, metadata)

        


if __name__ == '__main__':
    main()