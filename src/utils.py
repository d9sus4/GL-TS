"""
Utility functions.
"""

from tabulate import tabulate
import numpy as np


def print_metadata_legacy(metadata):
    print("-----Dataset metadata info-----")
    print(f"Dataset name: {metadata['name']}")
    print(f"Dataset size: {metadata['size']}")
    print(f"Sample interval: {metadata['sample_interval']}")
    print(f"Number of variables: {metadata['num_vars']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Average sequence length: {metadata['avg_seq_len']}")
    print(f"Sequence length range: [{metadata['min_seq_len']}, {metadata['max_seq_len']}]")
    print(f"Sequence length standard deviation: {metadata['seq_len_sigma']}\n")
    class_table = []
    for i in range(metadata['num_classes']):
        ratio = metadata['class_counts'][i] / metadata['size']
        class_table.append([i, metadata['class_names'][i], metadata['class_counts'][i], f"{ratio:.2%}"])
    print("Class table:")
    print(tabulate(class_table, headers=["", "Class name", "Class count", "Ratio"], tablefmt="grid"))
    sensor_table = []
    for i in range(metadata['num_sensors']):
        sensor_table.append([i, metadata['sensor_names'][i], f"{metadata['mean'][i]:.2f}", f"{metadata['std'][i]:.2f}", f"{metadata['missing_rates'][i]:.2%}"])
    print("\nSensor table:")
    print(tabulate(sensor_table, headers=["", "Sensor name", "Mean value", "Standard deviation", "Missing rate"], tablefmt="grid"))


def print_metadata(metadata):
    print("-----Dataset metadata info-----")
    print(f"Dataset name: {metadata['name']}")
    print(f"Description: {metadata['desc']}")
    print(f"Split: {metadata['split']}")
    print(f"Dataset size: {metadata['size']}")
    print(f"Imputation strategy: {metadata['impute_strat']}")
    print(f"Time unit: {metadata['time_unit']}")
    print(f"Sample interval: {metadata['sample_interval']}")
    print(f"Number of variables: {metadata['num_vars']}")
    print(f"Average sequence length: {metadata['mean_seq_len']}")
    print(f"Sequence length range: [{metadata['min_seq_len']}, {metadata['max_seq_len']}]")
    print(f"Sequence length standard deviation: {metadata['sigma_seq_len']}\n")
    var_table = []
    for i in range(metadata['num_vars']):
        var_table.append([i, metadata['var_names'][i], f"{metadata['mean'][i]:.2f}", f"{metadata['std'][i]:.2f}", f"{metadata['missing_rates'][i]:.2%}"])
    print("Variable table:")
    print(tabulate(var_table, headers=["", "Variable name", "Mean value", "Standard deviation", "Missing rate"], tablefmt="grid"))


def drop_vars_in_metadata(metadata, mask: np.ndarray):
    """Return metadata after dropping unwanted sensors.
    1's in `mask` will get preserved.
    """
    # metadata = metadata.copy()
    metadata['num_sensors'] = np.sum(mask)
    metadata['sensor_names'] = [name for name, keep in zip(metadata['sensor_names'], mask) if keep]
    metadata['mean'] = metadata['mean'][mask]
    metadata['std'] = metadata['std'][mask]
    metadata['missing_rates'] = metadata['missing_rates'][mask]
    metadata['is_categorical'] = metadata['is_categorical'][mask]