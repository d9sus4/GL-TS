# Green Learning for Modeling Irregularly Sampled Time Series with Missing Values

## 1 File Structure

- `./raw_data/`: a place to store usable datasets that fit our interest.
  
  - `mimic3/`: https://github.com/YerevaNN/mimic3-benchmarks
  - `p12/`: https://www.physionet.org/content/challenge-2012/1.0.0/
  - `p19/`: https://physionet.org/content/challenge-2019/1.0.0/
  - `pam/`: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
  
- `./data/`: all processed data are supposed to output here.

- `./src/`: all runable python codes.

  - `preprocess/`: codes for preprocessing, resampling, outliers removal and splitting.

    Always preprocess before imputation, etc.
    
    For every dataset, there's a corresponding script that preprocess the dataset into a bunch of `.pkl`s, usually `<dataset_name>_clean_<split>.pkl`.
    
    E.g., to preprocess P12 dataset, run `python -m src.preprocess.p12 --do-split` at root directory. 3 pickle files will be generated under `./data/p12/` and later experiments regarding imputation and classification can always go from there.
    
    See docstrings in each script to learn how to use them.

## 2 Dataset Specifications

### 2.1 P19

P19 is itself an hourly sampled dataset so no resampling is needed. However, for metrics, the original P19 has a utility function for scoring the prediction based on temporal precision. But here we treat it as a simple binary classification task, and **the binary labels are assigned based on the occurrence of sepsis**, aligning with what's done in [RAINDROP](https://arxiv.org/abs/2110.05357) ([code](https://github.com/mims-harvard/Raindrop)). So theoretically we can compare our results with what's reported by RAINDROP.

### 2.2 P12

P12 is an irregularly sampled time series database. It has multiple outcome descriptors but is mainly used for binary classification task such as in [GRU-D](https://arxiv.org/abs/1606.01865) ([code](https://github.com/PeterChe1990/GRU-D)), RAINDROP and [ViTST](https://arxiv.org/abs/2303.12799) ([code](https://github.com/Leezekun/ViTST)). The difference though, is that **GRU-D and ViTST both use mortality prediction as their primary task** (GRU-D also experimented on mortality + 3 other tasks, including los < 3d, as a multitask); while **RAINDROP uses length of stay < 3 day as the primary task**. (See [issue #14](https://github.com/Leezekun/ViTST/issues/14) under ViTST's repo). Note that, GRU-D does resample the time series into an hourly basis while RAINDROP and ViTST don't.

### 2.3 MIMIC-III

MIMIC-III is a massive dataset. GRU-D uses it for binary mortality prediction only. They use 17 specific features selected in [this benchmark work](https://www.sciencedirect.com/science/article/pii/S1532046418300716) ([code](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII/tree/dep_notebooks)).

