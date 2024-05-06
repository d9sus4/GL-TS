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
        All time series segments (episodes) will be saved in a HDF5 `out_dir/P19.hdf5` as a group `data` (under root) as individual groups with 4 datasets (key-value pairs) in each:
            `obs` (ndarray of float32): raw (non-normalized, categorical features are simply replaced as integer labels if any) sensor observation array.
                Shaped as (len, num_sensors).
                Missing values are NaNs.
            `stamp` (ndarray of float32): time stamps of size (len).
            `mask` (ndarray of boolean): missing observation masking, 1 for real observed value and 0 for missing value (maybe imputed later).
                Same shape as `obs`.
            `label` (int): classification label of the time series.
        A demographic info dictionary will be saved as a nested group `demogr` inside the segment group. In P19 there are 5 keys:
            `age` (int): 100 for patients aged 90 or above.
            `gender` (int): 0 for female and 1 for male.
            `unit_id` (int): 0 for micu, 1 for sicu and -1 for unknown.
            `hosp_adm_time` (float): t(icu admission) - t(hospital admission) in hours, usually negative.
            `icu_los` (float): length of icu stay in hours.
        Also, a metadata dictionary will also be saved to that HDF5 as a group `meta` under its root, providing some meta info about the processed dataset such as:
            `size` (int): size of the dataset.
            `num_sensors` (int): how many sensors (feature columns) are there.
            `num_classes` (int): how many classification classes are there.
            `sensor_names` (list of strings): names of the sensors, sized (num_sensors).
            `class_names` (list of strings): names of the classes, starting from 0 to num_classes-1.
            `class_counts` (ndarray of int32): number of samples in each class, sized (num_classes).
            `mean` (ndarray of float32): mean observed values in each sensor, shaped as (num_sensors).
            `std` (ndarray of float32): std of observed values in each sensor, shaped as (num_sensors).
            `missing_rates` (ndarray of float32): missing rates of each sensor modality, shaped as (num_sensors).
            `is_categorical` (ndarray of boolean): True if there is any categorical column (sensor).
            `sample_interval` (float32): dataset's constant sample interval in hour. Note that P19 is originally sampled at 1.0 hr.
                In case this term is negative or NaN, observations may not be regularly sampled, i.e. consecutive rows may not have same time intervals in between them.
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
from tabulate import tabulate

# from ..dataset import MyHDF5Handler


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


def process_psv(fp):
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

    # HDF5 file path
    hdf5_file_path = os.path.join(args.out_dir, 'P19.hdf5')
    os.makedirs(args.out_dir, exist_ok=True)

    psv_paths = glob(os.path.join(args.root_dir, 'training_set*/p*.psv'))
    print(f'Found {len(psv_paths)} psv files. Start processing.')

    # accumulators
    sum_obs = np.zeros(34, dtype=np.float64)
    sum_sq_obs = np.zeros(34, dtype=np.float64)
    count_obs = np.zeros(34, dtype=np.float64)
    count_nan_obs = np.zeros(34, dtype=np.float64)
    seq_lens = []

    # init metadata records
    metadata = {
        'name': 'P19',
        'size': 0, # to be counted later on the fly
        'num_sensors': 34,  # first 34 columns are sensors
        'num_classes': 2,  # binary classification for sepsis
        'sensor_names': [],  # to be filled later
        'class_names': ['Non-sepsis', 'Sepsis'],
        'class_counts': np.zeros(2, dtype=np.int32), # to be counted later on the fly
        'mean': None, # to be filled later
        'std': None, # to be filled later
        'missing_rates': None, # to be filled later
        'is_categorical': np.array([False] * 34),  # no categorical feature in P19
        'sample_interval': args.resample if args.resample is not None and args.resample > 0 else np.nan,
        'avg_seq_len': None, # to be filled later TODO
        'max_seq_len': None, # to be filled later TODO
        'min_seq_len': None, # to be filled later TODO
        'seq_len_sigma': None, # to be filled later TODO
    }

    # get column names
    example_df = pd.read_csv(psv_paths[0], sep='|')
    metadata['sensor_names'] = example_df.columns.tolist()[:34]

    # init HDF5
    with h5py.File(hdf5_file_path, 'w') as f:
        data_group = f.create_group('data')
        meta_group = f.create_group('meta')
        for i, fp in enumerate(tqdm(psv_paths, desc="Processing .psv files...")):
            try:
                obs, stamp, mask, label, demogr = process_psv(fp)
            except Exception as e:
                print(f'Something is off ({type(e)}) processing {fp}, skipping it.')
                continue

            seq_lens.append(len(stamp))
                
            # segment group name is the base filename without .psv
            base_name = os.path.splitext(os.path.basename(fp))[0]
            # create a group for each segment
            segment_group = data_group.create_group(base_name)
            # store each part of the segment as a dataset within the group
            for key, value in {'obs': obs, 'stamp': stamp, 'mask': mask, 'label': label}.items():
                segment_group.create_dataset(key, data=value)
            
            # store demographic info as a nested group `demogr`
            demogr_group = segment_group.create_group('demogr')
            for key, value in demogr.items():
                demogr_group.create_dataset(key, data=value)
            
            # count as a processed segment
            metadata['size'] += 1
            # add to class count
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
        
        # add metadata to meta group
        for key, value in metadata.items():
            meta_group.create_dataset(key, data=value)

    print(f'Processed {metadata["size"]} files and dataset saved to {hdf5_file_path}.')

    print("-----Dataset metadata info-----")
    print(f"Dataset name: {metadata['name']}")
    print(f"Dataset size: {metadata['size']}")
    print(f"Sample interval: {metadata['sample_interval']}")
    print(f"Number of sensors: {metadata['num_sensors']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Average sequence length: {metadata['avg_seq_len']}")
    print(f"Sequence length range: [{metadata['min_seq_len']}, {metadata['max_seq_len']}]")
    print(f"Sequence length sigma: {metadata['seq_len_sigma']}\n")
    class_table = []
    for i in range(metadata['num_classes']):
        ratio = metadata['class_counts'][i] / metadata['size']
        class_table.append([i, metadata['class_names'][i], metadata['class_counts'][i], f"{ratio:.2%}"])
    print("Classes info:")
    print(tabulate(class_table, headers=["", "Class name", "Class count", "Ratio"], tablefmt="grid"))
    sensor_table = []
    for i in range(metadata['num_sensors']):
        sensor_table.append([i, metadata['sensor_names'][i], f"{metadata['mean'][i]:.2f}", f"{metadata['std'][i]:.2f}", f"{metadata['missing_rates'][i]:.2%}"])
    print("\nSensors info:")
    print(tabulate(sensor_table, headers=["", "Sensor name", "Mean value", "Standard deviation", "Missing rate"], tablefmt="grid"))


if __name__ == '__main__':
    main()