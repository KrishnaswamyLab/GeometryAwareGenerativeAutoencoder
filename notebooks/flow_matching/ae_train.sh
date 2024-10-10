#!/bin/bash

conda activate yale529

python train_autoencoder.py \
    --ae_max_epochs 200 \
    --ae_early_stop_patience 50 \
    --latent_dim 3 \
    --batch_norm \
    --use_spectral_norm \
    --dropout 0.2 \
    --dist_mse_decay 0.0 \
    --weights_dist 77.4 \
    --weights_reconstr 0.32 \
    --weights_cycle 1 \
    --weights_cycle_dist 0 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --ambient_source pca \
    --test_group 2 \
    --data_save_dir ../../data/ \
    --data_path ../../data/multi_D-100_d-3_pca.npz \
    # --data_path ../../data/multi_D-100_d-3_pca.npz \
    # --data_path ../../data/cite_all_D-50_d-3_pca.npz \
    # --data_path ../../data/cite_all_D-100_d-3_pca.npz \
    # --data_path ../../data/cite_all_D-50_d-3_pca.npz \
    # --data_path ../../data/eb_subset_all.npz \
    # --data_path ../../data/cite_D-50_d-3_pca.npz \
    # --eb_h5ad_path ../../data/eb.h5ad \
    # --mode eval \