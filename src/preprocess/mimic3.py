"""
MIMIC-III In-Hospital Mortality Prediction Dataset - Preprocessing Script.
The raw data for the specific mortality prediction task is generated using mimic3-benchmarks: https://github.com/YerevaNN/mimic3-benchmarks.
TODO
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

import os
import sys
import time
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
import copy
import pickle
import itertools
from datetime import datetime
import pandas as pd
import heapq
import re
import math

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F

import torchvision
from torchvision import transforms

from tqdm import tqdm
import argparse

import glob
from datetime import datetime
from collections import defaultdict

from ..utils import print_metadata


FEATURE_LIST = [
        ("capillary_refill_rate", "0.0", "categorical"),
        ("diastolic_blood_pressure", 59.0, "continuous"),
        ("fraction_inspired_oxygen", 0.21, "continuous"),
        ("glascow_coma_scale_eye_opening", "4 spontaneously", "categorical"),
        ("glascow_coma_scale_motor_response", "6 obeys commands", "categorical"),
        ("glascow_coma_scale_total", "15", "categorical"),
        ("glascow_coma_scale_verbal_response", "5 oriented", "categorical"),
        ("glucose", 128.0, "continuous"),
        ("heart_rate", 86, "continuous"),
        ("height", 170.0, "continuous"),
        ("mean_blood_pressure", 77.0, "continuous"),
        ("oxygen_saturation", 98.0, "continuous"),
        ("respiratory_rate", 19, "continuous"),
        ("systolic_blood_pressure", 118.0, "continuous"),
        ("temperature", 36.6, "continuous"),
        ("weight", 81.0, "continuous"),
        ("ph", 7.4, "continuous"),
    ]
# form it into dictionary
FEATURE_DICT = {}
for name, avg, ftype in FEATURE_LIST:
    FEATURE_DICT[name] = {
        "avg": avg,
        "type": ftype,
        "num_cls": {
            "capillary_refill_rate": 2,
            "glascow_coma_scale_eye_opening": 4,
            "glascow_coma_scale_motor_response": 6,
            "glascow_coma_scale_total": 13,
            "glascow_coma_scale_verbal_response": 5,
        }.get(name, None),
        "bound": None,
    }
# for some columns that may contain abnormally big /small values, add boundaries
FEATURE_DICT["weight"]["bound"] = (0, 300)

def map_cat_label_str2int(name, value):
    """
    Map categorical string labels that contain certain substring to integer labels (starting from 0, np.nan for unknown)
    """

    if pd.isna(value):
        return np.nan
    substring_mapping = {
        # "capillary_refill_rate": 0.0 and 1.0
        "glascow_coma_scale_eye_opening": {
            "respon": 0,
            "pain": 1,
            "speech": 2,
            "spont": 3,
        },
        "glascow_coma_scale_motor_response": {
            "respon": 0,
            "extens": 1,
            "flex": 2,
            "withd": 3,
            "pain": 4,
            "obey": 5,
        },
        # "glascow_coma_scale_total": 3.0 to 15.0
        "glascow_coma_scale_verbal_response": {
            "respon": 0,
            "trach": 0,
            "incomp": 1,
            "inap": 2,
            "conf": 3,
            "orient": 4,
        },
    }
    if name == "capillary_refill_rate":
        return int(float(value))
    elif name == "glascow_coma_scale_total":
        return int(float(value)) - 3
    elif name in substring_mapping.keys():
        for substring, label in substring_mapping[name].items():
            if substring in value.lower():
                return label
        return np.nan
    else: # this is not a categorical column at all
        return value


def process_csv(fp):
    df = pd.read_csv(fp)
    # rename feature columns
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # get rid of out-of-boundary numeric values by replacing with nan
    for col_name in FEATURE_DICT.keys():
        if FEATURE_DICT[col_name]["type"] == "continuous" and FEATURE_DICT[col_name]["bound"] is not None:
            lo, hi = FEATURE_DICT[col_name]["bound"]
            df.loc[(df[col_name] < lo) | (df[col_name] > hi), col_name] = np.nan
    # deal with categorical features
    for col_name in FEATURE_DICT.keys():
        if FEATURE_DICT[col_name]["type"] == "categorical":
                df[col_name] = df[col_name].map(lambda x: map_cat_label_str2int(col_name, x))
    features_df = df.drop(columns='hours', inplace=False)
    time_stamps = df['hours']
    return features_df.to_numpy(dtype=float), time_stamps.to_numpy(dtype=float)



def preprocess(dir, out_path):
    """
    arguments:
        stats_dict: if not none, this stats_dict is used for normalizing
    """

    num_sensors = len(FEATURE_LIST)

    # init metadata records
    metadata = {
        'name': 'mimic3_ihm',
        'desc': 'Cleaned MIMIC-III Benchmark for in-hospital mortality prediction.',
        'split': 'train' if 'train' in dir else 'test',
        'time_unit': 'hour',
        'impute_strat': None,
        'size': 0, # to be counted later on the fly
        'num_vars': num_sensors,
        'var_names': np.array([x for x, _, _ in FEATURE_LIST]),
        'label_names': ['mortality'],
        'mean': None, # to be filled later
        'std': None, # to be filled later
        'missing_rates': None, # to be filled later
        'is_categorical': np.array([x == 'categorical' for _, _, x in FEATURE_LIST]),
        'sample_interval': np.nan, # no resampling
        'mean_seq_len': None, # to be filled later TODO
        'max_seq_len': None, # to be filled later TODO
        'min_seq_len': None, # to be filled later TODO
        'sigma_seq_len': None, # to be filled later TODO
    }

    # accumulators
    sum_obs = np.zeros(num_sensors, dtype=np.float64)
    sum_sq_obs = np.zeros(num_sensors, dtype=np.float64)
    count_obs = np.zeros(num_sensors, dtype=np.float64)
    count_nan_obs = np.zeros(num_sensors, dtype=np.float64)
    seq_lens = []
    
    all_data = []
    print(f"Start preprocessing data under {dir}")
    all_csv_paths = glob.glob(os.path.join(dir, "*_episode*_timeseries.csv"))
    all_data_frames = []
    all_keys = []
    all_labels = []
    # read listfile
    listfile_path = os.path.join(dir, "listfile.csv")
    listfile_dict = pd.read_csv(listfile_path, index_col=0).to_dict(orient="index")
    # read all dataframes into dfs, fill all nan cells and convert categorical to numeric
    print("Reading all time series csv files...")
    for i, path in enumerate(tqdm(all_csv_paths, desc="Processing csv files...")):
        try:
            # get the key
            re_match = re.match(r"(\d+)_episode(\d+)_timeseries.csv", os.path.basename(path))
            if not re_match:
                raise ValueError(f"Error parsing csv file: {path}")
            subject_id, episode_number = map(int, re_match.groups())
            key = f"{subject_id}_episode{episode_number}_timeseries.csv"

            # read the label
            if key not in listfile_dict.keys(): # check if labels are recorded in listfile
                raise KeyError(f"Mapping key not foound: {key}")
            label_dict = listfile_dict[key] # dict_keys(['length of stay', 'in-hospital mortality task (pos;mask;label)', 'length of stay task (masks;labels)', 'phenotyping task (labels)', 'decompensation task (masks;labels)'])
            label = label_dict['y_true']

            # read feature csv
            obs, time = process_csv(path)
            
            # create isnan mask
            mask = ~np.isnan(obs)

            interval = np.zeros_like(obs)
            for col in range(obs.shape[1]):
                last_observed_time = 0
                for row in range(obs.shape[0]):
                    interval[row, col] = time[row] - last_observed_time
                    if mask[row, col]: # observed here
                        last_observed_time = time[row]

        except Exception as e:
            print(f"Something is off ({type(e)}) when reading {key}, skipped it")
            continue

        # create data dict
        data_dict = {
            'id': key,
            'time': time,
            'var': obs,
            'mask': mask,
            'interval': interval,
            'label': label,
        }
        all_data.append(data_dict)
        seq_lens.append(len(time))
        # count as a processed episode
        metadata['size'] += 1

        # update sums for mean/std calculation
        sum_obs += np.where(mask, obs, 0).sum(axis=0)
        sum_sq_obs += np.where(mask, obs**2, 0).sum(axis=0)
        count_obs += mask.sum(axis=0)
        # count missing values
        count_nan_obs += (~mask).sum(axis=0)

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

    # save output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({
            'data': all_data,
            'meta': metadata,
        }, f)

    print(f'Processed {metadata["size"]} files and dataset saved to {out_path}.')
    print_metadata(metadata)


def main():
    preprocess('./raw_data/mimic3/mortality/train/', './data/mimic3/mimic3_mortality_train.pkl')
    preprocess('./raw_data/mimic3/mortality/test/', './data/mimic3/mimic3_mortality_test.pkl')


if __name__ == '__main__':
    main()