import argparse
import h5py  # Import h5py for handling HDF5 files

def parse_args():
    parser = argparse.ArgumentParser(description='Print metadata information from an HDF5 file.')
    parser.add_argument('fp', type=str, help='Path to the HDF5 file containing metadata.')
    return parser.parse_args()

def load_metadata(file_path):
    with h5py.File(file_path, 'r') as file:
        # Access the 'meta' group where metadata is stored
        meta_group = file['meta']
        metadata = {}
        for key in meta_group.keys():
            # For arrays or lists, convert them to a Python list for easier handling
            if len(meta_group[key].shape) > 0:
                metadata[key] = list(meta_group[key][:])
            else:  # For single elements, just take the value
                metadata[key] = meta_group[key][()]
            # Special handling for 'sensor_names' and 'class_names' due to being stored as bytes
            if key in ['sensor_names', 'class_names']:
                metadata[key] = [name.decode('utf-8') for name in metadata[key]]
    return metadata

def print_metadata(metadata):
    print("-----Dataset metadata info-----")
    print(f"Dataset size: {metadata['size']}")
    print(f"Number of sensors: {metadata['num_sensors']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Class names: {', '.join(metadata['class_names'])}")
    print(f"Sample interval: {metadata['sample_interval']}")
    print(f"Observed value statistics:")
    sensor_info = zip(metadata['sensor_names'], metadata['mean'], metadata['std'])
    print("{:<20} {:<15} {:<15}".format('Sensor Name', 'Mean', 'Std'))
    for sensor_name, mean, std in sensor_info:
        print("{:<20} {:<15} {:<15}".format(sensor_name, f"{mean:.2f}", f"{std:.2f}"))

if __name__ == '__main__':
    args = parse_args()
    metadata = load_metadata(args.fp)
    print_metadata(metadata)