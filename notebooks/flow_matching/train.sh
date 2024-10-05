#!/bin/bash

conda activate yale529

python geo_flow.py \
    --max_epochs 50 \
    --visualize_training \
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
    --show_plot \
    --embed_t \
    --plotly \
    --test_group 1 \
    --data_path ../../data/cite_D-50_d-3_pca.npz \
    --start_idx 736 \
    --end_idx 2543 \
    --start_group 0 \
    --end_group 2 \
    --range_size 0.3 \
    --use_all_group_points \
    --use_local_ae \
    --ae_checkpoint_path ./ae_checkpoints/autoencoder-v17.ckpt \
    --mode eval \

# python latent_discriminator.py \
#     --max_epochs 50 \
#     --visualize_training \
#     --neg_method add \
#     --num_samples 128 \
#     --batch_size 32 \
#     --noise_levels 0.5 1.0 \
#     --sampling_rejection \
#     --sampling_rejection_method sugar \
#     --sampling_rejection_threshold 0.2 \
#     --disc_batch_size 128 \
#     --disc_layer_widths 256 128 64 \
#     --disc_factor 10 \
#     --disc_max_epochs 100 \
#     --alpha 8.0 \
#     --fixed_pot \
#     --show_plot \
#     --embed_t \
#     --test_group 1 \
#     --data_path ../../data/cite_D-50_d-3_pca.npz \
#     --start_idx 736 \
#     --end_idx 2543 \
#     --start_group 0 \
#     --end_group 2 \
#     --range_size 0.3 \
#     --use_all_group_points \
#     --use_local_ae \
#     --ae_checkpoint_path ./ae_checkpoints/autoencoder-v17.ckpt \
#     --mode eval \
    # --sample_group_points \

    # --ae_checkpoint_path ./ae_checkpoints/autoencoder-v15.ckpt \
    # --mode eval \
    # --use_local_ae \
    # --disc_use_gp \

