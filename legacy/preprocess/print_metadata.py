import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Print metadata information from a metadata.pkl file.')
    parser.add_argument('metadata_file', type=str, help='Path to the metadata.pkl file.')
    return parser.parse_args()

def load_metadata(file_path):
    with open(file_path, 'rb') as file:
        metadata = pickle.load(file)
    return metadata

def print_metadata(metadata):
    print("-----Dataset metadata info-----")
    print(f"Dataset size: {metadata['size']}")
    print(f"Number of sensors: {metadata['num_sensors']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Class names: {', '.join(metadata['class_names'])}")
    print(f"Sample interval: {metadata['sample_interval']}")
    print(f"Observated value statistics:")
    sensor_info = zip(metadata['sensor_names'], metadata['mean'], metadata['std'])
    print("{:<20} {:<15} {:<15}".format('Sensor Name', 'Mean', 'Std'))
    for sensor_name, mean, std in sensor_info:
        print("{:<20} {:<15} {:<15}".format(sensor_name, f"{mean:.2f}", f"{std:.2f}"))

if __name__ == '__main__':
    args = parse_args()
    metadata = load_metadata(args.metadata_file)
    print_metadata(metadata)