# Installation
```sh
conda create -n dmae -c conda-forge python=3.11.5
conda activate dmae
pip install -r requirements.txt
pip install -e . # install the package in dev mode.
```
If you also want to use jupyter notebooks, install
```sh
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=dmae
```
# Getting Started:
## Example with toy swiss roll data (npy format)
- Generate the data: [notebooks/swiss_roll_data.ipynb](notebooks/swiss_roll_data.ipynb)
- run the model:
```sh
cd src
python main.py logger.use_wandb=False data.file_type=npy data.require_phate=False data.datapath=../data/swiss_roll.npy data.phatepath=../data/swiss_roll_phate.npy training.max_epochs=5
```
- check the results: [notebooks/swiss_roll_result.ipynb](notebooks/swiss_roll_result.ipynb)
## Example with BMMC myeloid data (anndata format)
- download the data from [https://github.com/KrishnaswamyLab/PHATE/blob/main/data/BMMC_myeloid.csv.gz](https://github.com/KrishnaswamyLab/PHATE/blob/main/data/BMMC_myeloid.csv.gz)
- prepare the data with [notebooks/myeloid_data.ipynb](notebooks/myeloid_data.ipynb)
- run the model:
```sh
cd src
python main.py logger.use_wandb=False data.file_type=h5ad data.require_phate=False data.datapath=../data/BMMC_myeloid.h5ad
```
- check the results: [notebooks/BMMC_myeloid_result.ipynb](notebooks/BMMC_myeloid_result.ipynb)

## Train Affinity Matching AE
- generate data
```sh
# Data Config file: af_data.yaml. The data will be saved in the '../data' folder.
python data_script.py

```
- train Affinity Matching AE. The decoder is trained separately from the encoder.
```sh
# example with overwriting the default parameters. Config file: separate_affinityae.yaml
python separate_affinityae.py logger.use_wandb=true data.name=swiss_roll model.prob_method=heat_kernel
```

## Geodesic Flow Matching with CITE Data in 100 PCA Dimension
- Train the model
```sh
cd notebooks/flow_matching
./train.sh
```
- Evaluate the model
```sh
cd notebooks/flow_matching
./eval.sh
```
