from dataset import MyHDF5Handler
import matplotlib.pyplot as plt
import numpy as np


def plot_imputed_timeseries(stamp, obs, mask, strat=None):
    seq_len, num_sensors = obs.shape
    
    # Generate a list of distinct colors
    colors = plt.cm.rainbow(np.linspace(0, 1, num_sensors))
    
    plt.figure(figsize=(20, 12))
    for sensor_idx in range(num_sensors):
        # Plot the full time series for this sensor
        plt.plot(stamp, obs[:, sensor_idx], label=f'Sensor {sensor_idx+1}', color=colors[sensor_idx])
        
        # Now, plot only the real observed values for this sensor
        real_obs_indices = np.where(mask[:, sensor_idx])[0]
        plt.scatter(stamp[real_obs_indices], obs[real_obs_indices, sensor_idx], color=colors[sensor_idx], marker='*', s=50)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Observation Value')
    plt.title(f'Imputed Time Series with Real Observations Marked{" (strategy = " + strat + ")" if strat is not None else ""}')
    plt.legend()
    plt.show()


def main():

    STRAT = ['spline', 'kmeans', 'xgboost'][2]
    EG_IDX = 666
    handler = MyHDF5Handler(f'../data/P19_{STRAT}.hdf5', read_only=True) # source dataset
    handler.print_metadata()

    metadata = handler.get_metadata()
    size = metadata['size']

    all_segment_keys = handler.get_all_segment_keys()

    segment_key = all_segment_keys[EG_IDX]
    segment = handler.get_segment(segment_key)
    obs = segment['obs']
    stamp = segment['stamp']
    mask = segment['mask']
    label = segment['label']
    demogr = segment['demogr']
    
    plot_imputed_timeseries(stamp, obs, mask, strat=STRAT)

if __name__ == '__main__':
    main()