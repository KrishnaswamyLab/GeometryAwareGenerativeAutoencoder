<h1 align="center">
[AISTATS 2025] GAGA 💃🪩
</h1>

<div align="center">
  <b>G</b>eometry-<b>A</b>ware <b>G</b>enerative <b>A</b>utoencoder for<br>
  Warped Riemannian Metric Learning and Generative Modeling on Data Manifolds
</div>

<div align="center">
  
[![ArXiv](https://img.shields.io/badge/ArXiv-GAGA-firebrick)](https://arxiv.org/abs/2410.12779)
[![AISTATS](https://img.shields.io/badge/AISTATS-lightgray)](https://proceedings.mlr.press/v258/sun25c.html)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social)](https://twitter.com/KrishnaswamyLab)
</div>


# Citation
If you find this work useful in your research, please consider citing:
```bibtex
@inproceedings{sun2025geometry,
  title={Geometry-Aware Generative Autoencoders for Warped Riemannian Metric Learning and Generative Modeling on Data Manifolds}, 
  author={Sun, Xingzhi and Liao, Danqi and MacDonald, Kincaid and Zhang, Yanlei and Liu, Chen and Huguet, Guillaume and Wolf, Guy and Adelstein, Ian and Rudner, Tim GJ and Krishnaswamy, Smita},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2025},
  organization={PMLR},      
}
```

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

# Visualizing the geometry-aware encoder
## Example with toy swiss roll data (npy format)
- Generate the data: [notebooks/swiss_roll_data.ipynb](notebooks/swiss_roll_data.ipynb)
- run the model:
```sh
cd src
python main.py \
    logger.use_wandb=False \
    data.file_type=npy \
    data.require_phate=False \
    data.datapath=../data/swiss_roll.npy \
    data.phatepath=../data/swiss_roll_phate.npy \
    training.max_epochs=5
```
- check the results: [notebooks/swiss_roll_result.ipynb](notebooks/swiss_roll_result.ipynb)
## Example with BMMC myeloid data (anndata format)
- download the data from [https://github.com/KrishnaswamyLab/PHATE/blob/main/data/BMMC_myeloid.csv.gz](https://github.com/KrishnaswamyLab/PHATE/blob/main/data/BMMC_myeloid.csv.gz)
- prepare the data with [notebooks/myeloid_data.ipynb](notebooks/myeloid_data.ipynb)
- run the model:
```sh
cd src
python main.py \
    logger.use_wandb=False \
    data.file_type=h5ad \
    data.require_phate=False \
    data.datapath=../data/BMMC_myeloid.h5ad
```
- check the results: [notebooks/BMMC_myeloid_result.ipynb](notebooks/BMMC_myeloid_result.ipynb)


# Transporting population for single-cell data
- Download the data from [kaggle](https://www.kaggle.com/competitions/open-problems-multimodal/data)
- Prepare the data with [notebooks/multi_data.ipynb](notebooks/multi_data.ipynb) and [notebooks/cite_data.ipynb](notebooks/cite_data.ipynb)
- Train the model (the example runs for CITE data in 100 PCA dimension)
```sh
cd notebooks/flow_matching
./train.sh
```
- Evaluate the model (the example runs for CITE data in 100 PCA dimension)
```sh
cd notebooks/flow_matching
./eval.sh
```
