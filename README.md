# Installation
```sh
conda create -n dmae -c conda-forge python=3.11.5
conda activate dmae
pip install -r requirements.txt
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

## Train Probability AE
- generate data
```sh
# example with swiss roll data. The data will be saved in the '../data' folder.
python data_script.py --data swiss_roll
```
- train the model
```sh
# example with overwriting the default parameters
python train_probae.py training.accelerator={cpu} training.max_epochs=5000
```