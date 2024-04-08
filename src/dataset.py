import h5py
import os

class MyHDF5Handler:
    def __init__(self, fp, read_only: bool):
        self.fp = fp
        self.read_only = read_only
        
        # Check if the file exists; if not, create the file and initial groups
        if not os.path.exists(fp) and not read_only:
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
        print(f"Dataset size: {metadata['size']}")
        print(f"Number of sensors: {metadata['num_sensors']}")
        print(f"Number of classes: {metadata['num_classes']}")
        print(f"Class names: {', '.join(metadata['class_names'])}")
        print(f"Sample counts in each class: {', '.join(str(x) for x in metadata['class_counts'])}")
        print(f"Sample interval: {metadata['sample_interval']}")
        print(f"Observed value statistics:")
        sensor_info = zip(metadata['sensor_names'], metadata['mean'], metadata['std'])
        print("{:<20} {:<15} {:<15}".format('Sensor Name', 'Mean', 'Std'))
        for sensor_name, mean, std in sensor_info:
            print("{:<20} {:<15} {:<15}".format(sensor_name, f"{mean:.2f}", f"{std:.2f}"))
    
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
