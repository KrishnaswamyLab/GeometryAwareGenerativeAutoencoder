# [AISTATS 2025] Geometry-Aware Generative Autoencoder (GAGA) 💃🪩
This is the code for the paper [Geometry-Aware Generative Autoencoder for Warped Riemannian
Metric Learning and Generative Modeling on Data Manifolds](https://arxiv.org/abs/2410.12779), AISTATS 2025.

# Citation
If you find this work useful in your research, please consider citing:
```bibtex
@misc{sun2025geometryawaregenerativeautoencoderswarped,
      title={Geometry-Aware Generative Autoencoders for Warped Riemannian Metric Learning and Generative Modeling on Data Manifolds}, 
      author={Xingzhi Sun and Danqi Liao and Kincaid MacDonald and Yanlei Zhang and Chen Liu and Guillaume Huguet and Guy Wolf and Ian Adelstein and Tim G. J. Rudner and Smita Krishnaswamy},
      year={2025},
      eprint={2410.12779},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.12779}, 
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


# Transporting population for Multi, CITE single-cell data
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
