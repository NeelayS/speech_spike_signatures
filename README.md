# speech_spike_signatures

Implementation of the paper [A Spiking Network that Learns to Extract Spike Signatures from Speech Signals](https://arxiv.org/abs/1606.00802) in PyGeNN: A Python Library for GPU-Enhanced Neural Networks

## Steps 

1. Install [PyGeNN](https://github.com/genn-team/genn/tree/master/pygenn).
2. Install dependencies in `requirements.txt`.
3. Download and extract the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset).
4. Run `src/preprocess.py` to pre-process the data and extract features from it.
5. Run `src/train.py` to simulate (train) the spiking neural network (SNN).

## Usage

### `preprocess.py`

To pre-process the data and perform feature extraction using the `preprocess.py` script -

```
usage: Script to pre-preocess the speech data [-h] --data_dir DATA_DIR [--data_file DATA_FILE] [--upper_bound UPPER_BOUND] [--lower_bound LOWER_BOUND]

required arguments:
  --data_dir DATA_DIR         Should point to the the folder containing all the audio .wav files

optional arguments:
  --data_file DATA_FILE       Name of the .npy file the pre-processed should be saved as (default='data')
  --upper_bound UPPER_BOUND   The upper bound of for scaling the feature extracted from the speech signal (default=52000.0)
  --lower_bound LOWER_BOUND   The lower bound of for scaling the feature extracted from the speech signal (default=52.0)
```

Example arguments -

`python preprocess.py --data_dir FSDD/recordings --data_file data --upper_bound 52000.0 --lower_bound 52.0`

<br/>

### `train.py`

To train (simulate) the SNN using the `train.py` script - 

```
usage: Script to train model the SNN using supervised STDP [-h] --datafile DATAFILE [--outdir OUTDIR] [--n_samples N_SAMPLES]

required arguments:
  --datafile DATAFILE        Path to the .npy file containing the speech data

optional arguments:
  --outdir OUTDIR            Name of folder where all the ouput files (membrane potentials etc) should be stored. Folder doesn't need to exist beforehand. (default='output')
  --n_samples N_SAMPLES      Number of samples in the dataset for which the network should be simulated. (default=1)
```

Example arguments -

`python train.py --datafile data/data_52000.npy --outdir results --n_samples 1`
