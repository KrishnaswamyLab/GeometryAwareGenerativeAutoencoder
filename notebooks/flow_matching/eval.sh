#!/bin/bash
# Evaluate the model on CITE data

conda activate dmae

python geo_flow.py \
    --max_epochs 10 \
    --neg_method add \
    --num_samples 128 \
    --batch_size 32 \
    --noise_levels 0.5 1.0 \
    --sampling_rejection \
    --sampling_rejection_method sugar \
    --sampling_rejection_threshold 0.2 \
    --disc_batch_size 128 \
    --disc_layer_widths 256 128 64 \
    --disc_factor 10 \
    --disc_max_epochs 100 \
    --alpha 8.0 \
    --fixed_pot \
    --embed_t \
    --start_group 0 \
    --end_group 2 \
    --test_group 1 \
    --range_size 0.3 \
    --use_all_group_points \
    --data_path ../../data/cite_all_D-100_d-3_pca.npz \
    --train_autoencoder \
    --ae_max_epochs 100 \
    --ae_early_stop_patience 50 \
    --ae_latent_dim 3 \
    --ae_batch_norm \
    --ae_use_spectral_norm \
    --ae_dropout 0.2 \
    --ae_dist_mse_decay 0.0 \
    --ae_weights_dist 77.4 \
    --ae_weights_reconstr 0.32 \
    --ae_weights_cycle 1 \
    --ae_weights_cycle_dist 0 \
    --ae_lr 1e-3 \
    --ae_weight_decay 1e-4 \
    --ambient_source pca \
    --autoencoder_ckptname autoencoder.ckpt \
    --discriminator_ckptname discriminator.ckpt \
    --gbmodel_ckptname gbmodel.ckpt \
    --mode eval \

