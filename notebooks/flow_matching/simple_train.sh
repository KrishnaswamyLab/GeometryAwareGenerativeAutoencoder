#!/bin/bash

conda activate yale529

python simple_gfm.py \
    --max_epochs 1 \
    --visualize_training \
    --neg_method add \
    --num_samples 128 \
    --batch_size 8 \
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
    --start_idx 736 \
    --end_idx 2543 \
    --start_group 0 \
    --end_group 2 \
    --range_size 0.3 \
    --use_all_group_points \
    --data_path ../../data/cite_all_D-1000_d-3_hvg.npz \
    --ae_max_epochs 200 \
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
    --ambient_source hvg \
    --ae_use_pretrained \
    --autoencoder_ckptname autoencoder.ckpt \
    --disc_use_pretrained \
    --discriminator_ckptname discriminator.ckpt \
    # --autoencoder_ckptname autoencoder-v1.ckpt \
    # --discriminator_ckptname discriminator-v1.ckpt \
    # --gbmodel_ckptname gbmodel.ckpt \
    #--train_autoencoder \
    # --mode eval \
    # --use_local_ae \
    # --ae_checkpoint_dir ./ae_checkpoints \
    # --autoencoder_ckptname autoencoder-v1.ckpt \
    # --discriminator_ckptname discriminator-v1.ckpt \
    # --gbmodel_ckptname gbmodel.ckpt \
    # --mode eval \


    # --data_path ../../data/eb_subset_all.npz \

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

