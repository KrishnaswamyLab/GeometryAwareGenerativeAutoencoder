# Installation
```sh
conda create -n dmae -c conda-forge python=3.11.5
conda activate dmae
pip install -r requirements.txt
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=dmae
```