# Green learning for modeling irregularly sampled time series with missingness

## Files

- `raw_data/`: a place to store usable datasets that fit our interest.
  - `MIMIC3/`: https://github.com/YerevaNN/mimic3-benchmarks
  - `P12/`: https://www.physionet.org/content/challenge-2012/1.0.0/
  - `P19/`: https://physionet.org/content/challenge-2019/1.0.0/
  - `PAM/`: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

- `src/`: all runable python codes.

  - `preprocess/`: codes for preprocessing.

    For every dataset, there's a corresponding script that preprocess the dataset into a bunch of resampled `.pkl`s and a meta file. Refer to the code docstrings.