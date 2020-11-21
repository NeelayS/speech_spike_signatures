# speech_spike_signatures

## Usage

To train (simulate) the SNN using the ```train.py``` script - 

```
usage: Script to train model the SNN using supervised STDP [-h] --datafile DATAFILE [--outdir OUTDIR] [--n_samples N_SAMPLES]

required arguments:
  --datafile DATAFILE        Path to the .npy file containing the speech data

optional arguments:
  --outdir OUTDIR            Name of folder where all the ouput files (membrane potentials etc) should be stored. Folder doesn't need to exist beforehand. (default = output)
  --n_samples N_SAMPLES      Number of samples in the dataset for which the network should be simulated. (default=1)
```

Example arguments -

`python train.py --datafile data/data_52000.npy --outdir results --n_samples 1`