import h5py
import os
from tabulate import tabulate
import argparse
import numpy as np
import pickle
from torch import nn
from torch.utils.data import Dataset

class MyHDF5Handler:
    def __init__(self, fp, read_only:bool=False, overwrite:bool=False):
        self.fp = fp
        self.read_only = read_only
        self.overwrite = overwrite
        
        # Check if the file exists; if not, create the file and initial groups
        if (not os.path.exists(fp) and not read_only) or (os.path.exists(fp) and overwrite):
            with self._open_file(mode='w') as f:
                f.create_group('data')
                f.create_group('meta')
        elif not os.path.exists(fp) and read_only:
            raise FileNotFoundError(f"No such file: '{fp}'")
    
    def _open_file(self, mode=None):
        """Utility method to open the HDF5 file."""
        if mode is None:
            mode = 'r' if self.read_only else 'a'
        return h5py.File(self.fp, mode)
    
    def get_all_segment_keys(self):
        """Read and return all keys (names) of segments."""
        with self._open_file() as f:
            data_group = f['data']
            segment_keys = list(data_group.keys())
        return segment_keys
    
    def get_segment(self, key):
        """Get a segment's object by key (name or index).
        Note that if key is an index, this function may run ~10x slower."""
        # segment_keys = self.get_all_segment_keys()
        if isinstance(key, int):
            segment_keys = self.get_all_segment_keys()
            segment_key = segment_keys[key]  # treat key as an index
        elif isinstance(key, str):
            segment_key = key  # treat key as a name
        else:
            raise ValueError("id must be either an integer (index) or a string (dataset name)")
        
        with self._open_file() as f:
            data_group = f['data'][segment_key]
            segment = {
                'obs': data_group['obs'][:],
                'stamp': data_group['stamp'][:],
                'mask': data_group['mask'][:],
                'label': data_group['label'][()],
                'demogr': {k: v[()] for k, v in data_group['demogr'].items()}
            }

        return segment
    
    def get_metadata(self):
        """Get the meta info of the dataset."""
        with self._open_file() as f:
            meta_group = f['meta']
            metadata = {key: meta_group[key][()] for key in meta_group.keys()}
            for key in ['sensor_names', 'class_names',]:
                metadata[key] = [name.decode('utf-8') for name in metadata[key]]
        return metadata
    
    def print_metadata(self):
        metadata = self.get_metadata()
        print("-----Dataset metadata info-----")
        print(f"Dataset name: {metadata['name']}")
        print(f"Dataset size: {metadata['size']}")
        print(f"Sample interval: {metadata['sample_interval']}")
        print(f"Number of sensors: {metadata['num_sensors']}")
        print(f"Number of classes: {metadata['num_classes']}")
        print(f"Average sequence length: {metadata['avg_seq_len']:.2f}")
        print(f"Sequence length range: [{metadata['min_seq_len']}, {metadata['max_seq_len']}]")
        print(f"Sequence length sigma: {metadata['seq_len_sigma']:.2f}\n")
        class_table = []
        for i in range(metadata['num_classes']):
            ratio = metadata['class_counts'][i] / metadata['size']
            class_table.append([i, metadata['class_names'][i], metadata['class_counts'][i], f"{ratio:.2%}"])
        print("Classes info:")
        print(tabulate(class_table, headers=["", "Class name", "Class count", "Ratio"], tablefmt="grid"))
        sensor_table = []
        for i in range(metadata['num_sensors']):
            sensor_table.append([i, metadata['sensor_names'][i], f"{metadata['mean'][i]:.2f}", f"{metadata['std'][i]:.2f}", f"{metadata['missing_rates'][i]:.2%}", 'Categorical' if metadata['is_categorical'][i] else 'Continuous'])
        print("\nSensors info:")
        print(tabulate(sensor_table, headers=["", "Sensor name", "Mean value", "Standard deviation", "Missing rate", "Type"], tablefmt="grid"))
    
    @staticmethod
    def drop_sensors(metadata, mask: np.ndarray):
        """Return metadata after dropping selected sensors."""
        metadata = metadata.copy()
        metadata['num_sensors'] = np.sum(mask)
        metadata['sensor_names'] = [name for name, keep in zip(metadata['sensor_names'], mask) if keep]
        metadata['mean'] = metadata['mean'][mask]
        metadata['std'] = metadata['std'][mask]
        metadata['missing_rates'] = metadata['missing_rates'][mask]
        metadata['is_categorical'] = metadata['is_categorical'][mask]
        return metadata

    def add_segment(self, key, data_dict):
        """Add the data_dict into data group, with the dataset name defined as key."""
        if self.read_only:
            raise ValueError("The HDF5 file is opened in read-only mode.")
        with self._open_file() as f:
            segment_group = f['data'].create_group(key)
            for data_key, value in data_dict.items():
                if data_key != 'demogr':
                    segment_group.create_dataset(data_key, data=value)
                else:  # Handle demographic info as a nested group
                    demogr_group = segment_group.create_group('demogr')
                    for k, v in value.items():
                        demogr_group.create_dataset(k, data=v)
    
    def set_metadata(self, meta_dict):
        """Clear existing meta group and rewrite it with key-values in meta_dict."""
        if self.read_only:
            raise ValueError("The HDF5 file is opened in read-only mode.")
        with self._open_file() as f:
            # Remove the existing 'meta' group if it exists
            if 'meta' in f:
                del f['meta']
            meta_group = f.create_group('meta')
            for key, value in meta_dict.items():
                meta_group.create_dataset(key, data=value)

def h5_to_pkl(fp):
    handler = MyHDF5Handler(fp)
    metadata = handler.get_metadata()
    print(metadata)
    all_keys = handler.get_all_segment_keys()
    print(len(all_keys))
    seg_example = handler.get_segment(all_keys[0])
    print(all_keys[0])
    print(seg_example.keys())
    ds_dict = {
        'meta': metadata,
        'data': [],
    }
    for key in all_keys:
        data_dict = handler.get_segment(key)
        data_dict['name'] = key
        ds_dict['data'].append(data_dict)
    print(len(ds_dict['data']))
    ds_dir, ds_name = os.path.split(fp)
    new_name = os.path.splitext(ds_name)[0] + '.pkl'
    new_path = os.path.join(ds_dir, new_name)
    with open(new_path, 'wb') as f:
        pickle.dump(ds_dict, f)


class MTSDataset(Dataset):
    def __init__(self):
        pass





def main():
    parser = argparse.ArgumentParser(description='Impute a preprocessed multivariate time series (missing values included) dataset.')
    parser.add_argument('--fp', type=str, default='../data/P19.hdf5',
                        help='Path to the processed dataset hdf5 file.')
    args = parser.parse_args()
    handler = MyHDF5Handler(args.fp, read_only=True)
    handler.print_metadata()



if __name__ == '__main__':
    # main()
    h5_to_pkl('../data/P19_linear.hdf5')