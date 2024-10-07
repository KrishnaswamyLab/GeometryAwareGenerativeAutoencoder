import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import phate
import scanpy as sc

def main():
    # Meta data
    DATA_DIR = "../../data/multi_cite/"
    FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")
    FP_TRAIN_MULTI_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")

    df_meta = pd.read_csv(FP_CELL_METADATA)
    print(df_meta.head())

    # Read train data
    df_list = []
    chunk_size = 2000
    total_rows = 105942
    for start in range(0, total_rows, chunk_size):
        df_train = pd.read_hdf(FP_TRAIN_MULTI_INPUTS, start=start, stop=start+chunk_size)
        df_list.append(df_train)
    # df_train = pd.concat(df_list)
    # print(df_train.head())

    # Donor 
    donor = 13176
    days_to_keep = [2, 3, 4, 7]
    df_donor_meta = df_meta[(df_meta['donor']==donor) & (df_meta['technology']=='multiome') & (df_meta['day'].isin(days_to_keep))]
    print(df_donor_meta.shape)

    # Get donor cell ids
    donor_cell_ids = df_donor_meta['cell_id'].tolist()
    print('len(donor_cell_ids) of multiome:', len(donor_cell_ids))

    # Filter train data based on the cell barcode
    df_train_donor_list = []
    for df in df_list:
        df_train_donor_list.append(df[df.index.isin(donor_cell_ids)])
    df_train_donor = pd.concat(df_train_donor_list)
    print('df_train_donor.shape:', df_train_donor.shape)

    # Subsample train data
    sample_size = 28000
    df_train_donor = df_train_donor.sample(n=sample_size, random_state=42)
    print('df_train_donor.shape:', df_train_donor.shape)

    adata = ad.AnnData(df_train_donor)
    print('adata:', adata)
    # Highly variable genes
    # sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    
    # PCA
    sc.pp.pca(adata, n_comps=50)

    # adata.var_names = df_train_donor.columns
    adata.obs_names = df_train_donor.index
    # adata.obs['donor'] = donor
    # adata.obs['technology'] = 'multiome'
    # adata.obs['cell_id'] = df_train_donor.index
    # adata.var['feature_name'] = df_train_donor.columns
    adata.write(f"adata_multi_train_donor-13176_size-{sample_size}_processed.h5ad")

    print('Done!')

if __name__ == "__main__":
    main()